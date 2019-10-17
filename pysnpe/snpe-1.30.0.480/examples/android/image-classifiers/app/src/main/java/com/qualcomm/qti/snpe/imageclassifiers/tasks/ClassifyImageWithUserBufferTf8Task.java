/*
 * Copyright (c) 2018 Qualcomm Technologies, Inc.
 * All Rights Reserved.
 * Confidential and Proprietary - Qualcomm Technologies, Inc.
 */
package com.qualcomm.qti.snpe.imageclassifiers.tasks;

import android.graphics.Bitmap;
import android.os.SystemClock;
import android.util.Pair;

import com.qualcomm.qti.snpe.NeuralNetwork;
import com.qualcomm.qti.snpe.TF8UserBufferTensor;
import com.qualcomm.qti.snpe.TensorAttributes;
import com.qualcomm.qti.snpe.UserBufferTensor;
import com.qualcomm.qti.snpe.imageclassifiers.Model;
import com.qualcomm.qti.snpe.imageclassifiers.ModelOverviewFragmentController;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.HashMap;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;

public class ClassifyImageWithUserBufferTf8Task extends AbstractClassifyImageTask {

    private static final String LOG_TAG = ClassifyImageWithUserBufferTf8Task.class.getSimpleName();

    private static final int TF8_SIZE = 1;

    private static final int TF8_BITWIDTH = 8;

    private static final int mStepExactly0 = 0;

    private static final float mStepSize = 1.0f;

    public ClassifyImageWithUserBufferTf8Task(ModelOverviewFragmentController controller,
                                              NeuralNetwork network, Bitmap image, Model model) {
        super(controller, network, image, model);
    }

    @Override
    protected String[] doInBackground(Bitmap... params) {
        final List<String> result = new LinkedList<>();

        final Map<String, TF8UserBufferTensor> inputTensors = new HashMap<>();
        final Map<String, TF8UserBufferTensor> outputTensors = new HashMap<>();

        final Map<String, ByteBuffer> inputBuffers = new HashMap<>();
        final Map<String, ByteBuffer> outputBuffers = new HashMap<>();

        boolean status = prepareInputs(inputTensors, inputBuffers);
        if (!status) {
            return new String[0];
        }
        prepareOutputs(outputTensors, outputBuffers);

        final long javaExecuteStart = SystemClock.elapsedRealtime();
        status = mNeuralNetwork.execute(inputTensors, outputTensors);
        final long javaExecuteEnd = SystemClock.elapsedRealtime();
        if (!status) {
            return new String[0];
        }
        mJavaExecuteTime = javaExecuteEnd - javaExecuteStart;

        float[] outputValues = dequantize(outputTensors.get(mOutputLayer), outputBuffers.get(mOutputLayer));
        for (Pair<Integer, Float> pair : topK(1, outputValues)) {
            result.add(mModel.labels[pair.first]);
            result.add(String.valueOf(pair.second));
        }

        String[] resultString = result.toArray(new String[result.size()]);

        releaseTensors(inputTensors, outputTensors);

        return resultString;
    }

    private boolean prepareInputs(final Map<String, TF8UserBufferTensor> inputTensors,
                                  final Map<String, ByteBuffer> inputBuffers) {
        TensorAttributes inputAttributes = mNeuralNetwork.getTensorAttributes(mInputLayer);
        Tf8Params inputParams = resolveTf8Params(inputAttributes);

        inputBuffers.put(mInputLayer, ByteBuffer.allocateDirect(inputParams.size).order(ByteOrder.nativeOrder()));

        loadMeanImageIfAvailable(mModel.meanImage, inputParams.size);

        final int[] dimensions = inputAttributes.getDims();
        final boolean isGrayScale = (dimensions[dimensions.length -1] == 1);
        float[] imageBitmapAsFloat;
        if (!isGrayScale) {
            imageBitmapAsFloat = loadRgbBitmapAsFloat(mImage);
        } else {
            imageBitmapAsFloat = loadGrayScaleBitmapAsFloat(mImage);
        }
        quantize(imageBitmapAsFloat, inputBuffers.get(mInputLayer), inputParams);

        inputTensors.put(mInputLayer, mNeuralNetwork.createTF8UserBufferTensor(
                inputParams.size, inputParams.strides,
                inputParams.stepExactly0, inputParams.stepSize,
                inputBuffers.get(mInputLayer)));

        return true;
    }

    private void prepareOutputs(final Map<String, TF8UserBufferTensor> outputTensors,
                                final Map<String, ByteBuffer> outputBuffers) {
        TensorAttributes outputAttributes = mNeuralNetwork.getTensorAttributes(mOutputLayer);
        Tf8Params outputParams = resolveTf8Params(outputAttributes);
        outputParams.stepExactly0 = mStepExactly0;
        outputParams.stepSize = mStepSize;

        outputBuffers.put(mOutputLayer, ByteBuffer.allocateDirect(outputParams.size).order(ByteOrder.nativeOrder()));
        outputTensors.put(mOutputLayer, mNeuralNetwork.createTF8UserBufferTensor(
                outputParams.size, outputParams.strides,
                outputParams.stepExactly0, outputParams.stepSize,
                outputBuffers.get(mOutputLayer)));
    }

    @SafeVarargs
    private final void releaseTensors(Map<String, ? extends UserBufferTensor>... tensorMaps) {
        for (Map<String, ? extends UserBufferTensor> tensorMap: tensorMaps) {
            for (UserBufferTensor tensor: tensorMap.values()) {
                tensor.release();
            }
        }
    }

    private void quantize(float[] src, ByteBuffer dst, Tf8Params tf8Params) {
        Tf8Encoding encoding = getTf8Encoding(src);

        byte[] quantized = new byte[src.length];
        for (int i = 0; i < src.length; i++) {
            float data = Math.max(Math.min(src[i], encoding.max), encoding.min);
            data = data / encoding.delta - encoding.offset;
            quantized[i] = (byte) Math.round(data);
        }

        dst.put(quantized);
        tf8Params.stepSize = encoding.delta;
        tf8Params.stepExactly0 = Math.round(-encoding.min / encoding.delta);
    }

    private Tf8Encoding getTf8Encoding(float[] array) {
        Tf8Encoding encoding = new Tf8Encoding();

        int num_steps = (int) Math.pow(2, TF8_BITWIDTH) - 1;
        float new_min = Math.min(getMin(array), 0);
        float new_max = Math.max(getMax(array), 0);

        float min_range = 0.1f;
        new_max = Math.max(new_max, new_min + min_range);
        encoding.delta = (new_max - new_min) / num_steps;

        if (new_min < 0 && new_max > 0) {
            float quantized_zero = Math.round(-new_min / encoding.delta);
            quantized_zero = (float) Math.min(num_steps, Math.max(0.0, quantized_zero));
            encoding.offset = -quantized_zero;
        } else {
            encoding.offset = Math.round(new_min / encoding.delta);
        }

        encoding.min = encoding.delta * encoding.offset;
        encoding.max = encoding.delta * num_steps + encoding.min;

        return encoding;
    }

    private float[] dequantize(TF8UserBufferTensor tensor, ByteBuffer buffer) {
        final int outputSize = buffer.capacity();
        final byte[] quantizedArray = new byte[outputSize];
        buffer.get(quantizedArray);

        final float[] dequantizedArray = new float[outputSize];
        for (int i = 0; i < outputSize; i++) {
            int quantizedValue = (int)quantizedArray[i] & 0xFF;
            dequantizedArray[i] = tensor.getMin() + quantizedValue *  tensor.getQuantizedStepSize();
        }

        return dequantizedArray;
    }

    private Tf8Params resolveTf8Params(TensorAttributes attribute) {
        int rank = attribute.getDims().length;
        int[] strides = new int[rank];
        strides[rank - 1] = TF8_SIZE;
        for (int i = rank - 1; i > 0; i--) {
            strides[i-1] = strides[i] * attribute.getDims()[i];
        }

        int bufferSize = TF8_SIZE;
        for (int dim: attribute.getDims()) {
            bufferSize *= dim;
        }

        return new Tf8Params(bufferSize, strides);
    }

    private class Tf8Params {
        int size;
        int[] strides;
        int stepExactly0;
        float stepSize;

        Tf8Params(int size, int[] strides) {
            this.size = size;
            this.strides = strides;
        }
    }

    private class Tf8Encoding {
        float min;
        float max;
        float delta;
        float offset;
    }

}
