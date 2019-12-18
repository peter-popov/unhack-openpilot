# Reverse engineer openpilot

My attempt to understand how smart the openpilot is. Inspired by this blogpost: [**How open pilot works(2017)**](https://medium.com/@comma_ai/how-does-openpilot-work-c7076d4407b3). Which is till somehow relevant, but misses all the interesting details about the vision system.

## Image preprocessing

Before sending to the model image taken from the camera goes trough a few pre-reprocessing steps. There are a lot of undocumented low-level opencl code to transform the image. Super difficult to understand what is happening there 😱 Looks like they do two main steps:

### Warp Perspective

Transform image to a road plane  (transform.c)

![](https://paper-attachments.dropbox.com/s_AF863A6EE917F218080502F04BD26B13583D2C3C86E286EBB657D5150557D7BB_1571588209660_frame.png)
![](https://paper-attachments.dropbox.com/s_AF863A6EE917F218080502F04BD26B13583D2C3C86E286EBB657D5150557D7BB_1571588285436_warp.png)

### Convert color space

They converted to [YUV](https://en.wikipedia.org/wiki/YUV) 4:2:0 (vision.cc). YUV420 has a different dimensions from different channels(U and V are half of Y), it took me a while to understand how do the give to NN which must expect all channels to be of the same size. I think after this they do the channels transformation as described in this [Efficient Pre-Processor for CNN paper](https://www.ingentaconnect.com/contentone/ist/ei/2017/00002017/00000019/art00009?crawler=true&mimetype=application/pdf) 

![](https://paper-attachments.dropbox.com/s_AF863A6EE917F218080502F04BD26B13583D2C3C86E286EBB657D5150557D7BB_1570825285017_Screenshot+from+2019-10-11+22-20-51.png)
![](https://paper-attachments.dropbox.com/s_AF863A6EE917F218080502F04BD26B13583D2C3C86E286EBB657D5150557D7BB_1570825270233_Screenshot+from+2019-10-11+22-20-31.png)


I cannot confirm the last step, as it’s very hard to understand the opencl code. But the fact that they ended up with **6x128x256 input** layers for their CNN, makes it very likely. Main purpose is to reduce computation on the device. Also my experiment kind of confirms this, as I managed to get a proper result our of their model after doing this channel trick.

## Deep Learning SDK

They use [Qualcomm Neural Processing SDK](https://developer.qualcomm.com/software/qualcomm-neural-processing-sdk) - deep learning library optimized for Snapdragon chips. As I understand it can do only inference. It stores the models in the proprietary *.dlc format.

The SDK allows importing models from Caffe, ONNX and TF. **But does not allows exporting into another formats!** Essentially this means that stealing their model is not very straight forward. When you think about it, it kind of makes sense. The SDK is supposed to be used on smartphones, where the model is shipped as an resource within an apk. And essentially can be taken by anyone.
I found an interesting [talk](https://www.youtube.com/watch?v=Dn3jb2BBBCE
) and [slides](https://conference.hitb.org/hitbsecconf2018dxb/materials/D1T1%20-%20AI%20Model%20Security%20-%20Reverse%20Engineering%20Machine%20Learning%20Models%20-%20Kang%20Li.pdf) on DL model security.

I have written a simple python wrapper around the SNPE model SDK so that I can use it from jupyter.

## DL Model

It looks very custom made. Open [html file](driving-model.html) for more details.

### Model layers

Some kind of resnet/inception for the encoder with 1 dense layer on top produces 1x512 feature vector.

The last concatenated with 1x8 desire vector. Not sure what it is, they always pass NULL at the moment. My guess it’s related to the route planner, e.g. telling model where you want to go 🛣️ 

That extended 1x520 feature vector goes into LSTM (it’s optional but enabled in the code at the moment).

Output of LSTM goes into 4 separate dense 3-layer [MDN](https://mikedusenberry.com/mixture-density-networks)s. The road coordinate system: x→forward, y→left, z→up. They predict prob distribution of lane $$y$$ coordinate, when $$x$$ goes from 1 to 192. 


### Output vector

| Number of elements | Meaning                                 |
| ------------------ | --------------------------------------- |
| 192                | Y coordinate of the predicted **path**  |
| 192                | std of each coordinate from above       |
| 192                | Y coordinate of the **left lane**       |
| 192                | std of each point above                 |
| 1                  | confidence of the left lane prediction  |
| 192                | Y coordinate of the **right lane**      |
| 192                | std of above each point above           |
| 1                  | confidence of the right lane prediction |
| 58                 | **Lead car** prediction, see below      |
| 512                | LSTM cell state                         |

I think I managed to receive a resonable result from the model. Expect that path prediction does not work at all(perhaps it's need RNN input):

![Predicted points of lanes and path(green) for the frame above](https://paper-attachments.dropbox.com/s_AF863A6EE917F218080502F04BD26B13583D2C3C86E286EBB657D5150557D7BB_1571589783975_prediction.png)

The lead(last 1x58 output) seems to be a [MDN](https://mikedusenberry.com/mixture-density-networks) of size 5 which estimates the location of the can we are following.

      // Every output distribution from the MDN includes the probabilties
      // of it representing a current lead car, a lead car in 2s
      // or a lead car in 4s

They use the following structure:

| 0             | dist                | distance to the car (0, 140.0)                                        |
| ------------- | ------------------- | --------------------------------------------------------------------- |
| 1             | y                   | I guess it’s a horizontal offset                                      |
| 2             | v                   | Relative velocity (0, 10.0)                                           |
| 3             | a                   | Angle I assume                                                        |
| 4             | std(dist), softplus |                                                                       |
| 5             | std(y), softplus    |                                                                       |
| 6             | std(v), softplus    |                                                                       |
| 7             | std(a), softplus    |                                                                       |
| 8             | ?                   | Mixture params. Gausian with max field is used for the lead_car       |
| 9             | ?                   | Mixture params. Gausian with max field is used for the furute_ead_car |
| 10            | ?                   |                                                                       |
| 5x above      |                     |                                                                       |
| 55            | prob lead           | ?                                                                     |
| 56            | prod future lead    | ?                                                                     |
| 57            | ?                   | ?                                                                     |

I haven’t seen it being used anywhere except visualization in UI module.

## Planning

Planning is split in 2 parts: longitudinal(planner.py) and lateral(pathplaner.py, lane_planner.py).

Points and stds from the model are used to fit 4 degree polynomial for path, left and right lane. As I can see only polynomials are later used for path planning.

### Longitudinal

Longitudinal planner provides velocity and acceleration. There seems to be 4 sources for predicting those:

1. Cruise control of the car
2. Path predicted by the DL model. Curvature used to estimate maximum feasible speed
3. Two different [MPC](https://en.wikipedia.org/wiki/Model_predictive_control) solvers. Not sure what is the difference between them

The final controls are chosen using some kind of heuristic. Seems to prefer the “slowest” model, which makes sense for safety and comfort reasons.

### Lateral

Choosing the path to follow:
$$L, R, P$$ - polynomial for the left and right lane, and path as predicted by the model
$$p_L, p_R, p_P$$ - probability of the left and right lane as predicted by the model

$$D=P_L\cdot P_R\frac{P_L\cdot L+P_R\cdot R}{P_L+P_R+0.0001} + (1-P_L\cdot P_R)\cdot P$$

The result is transmitted to the control module as a PathPlan. There is a lot of code which calculates accelerations, steering angles and checks braking. Very complex and hard to read. In general it seems that control module will execute the path via [PID](https://en.wikipedia.org/wiki/PID_controller), [LQR](https://en.wikipedia.org/wiki/Linear%E2%80%93quadratic_regulator) or INDI controller.

## Run the code

    > make notebook 

Takes some time to build a docker image

## todo

1. Try multiple frames to test what LSTM is doing. I think this is needed to get a proper path prediction from the model.
2. They work on lane change logic according to [this video](https://youtu.be/GiS68Uf_zsI). There seems to be a place in the network to ingest an intent. I am curious to try it when they release it.
