/*
 * Copyright (c) 2016-2018 Qualcomm Technologies, Inc.
 * All Rights Reserved.
 * Confidential and Proprietary - Qualcomm Technologies, Inc.
 */
package com.qualcomm.qti.snpe.imageclassifiers.tasks;

import android.graphics.Bitmap;
import android.os.AsyncTask;
import android.util.Pair;

import com.qualcomm.qti.snpe.NeuralNetwork;
import com.qualcomm.qti.snpe.imageclassifiers.Model;
import com.qualcomm.qti.snpe.imageclassifiers.ModelOverviewFragmentController;

import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;
import java.util.Set;

public abstract class AbstractClassifyImageTask extends AsyncTask<Bitmap, Void, String[]> {

    private static final String LOG_TAG = AbstractClassifyImageTask.class.getSimpleName();

    private static final int FLOAT_SIZE = 4;

    final String mInputLayer;

    final String mOutputLayer;

    private final ModelOverviewFragmentController mController;

    final NeuralNetwork mNeuralNetwork;

    final Model mModel;

    final Bitmap mImage;

    private FloatBuffer mMeanImage;

    long mJavaExecuteTime = -1;

    AbstractClassifyImageTask(ModelOverviewFragmentController controller,
                                     NeuralNetwork network, Bitmap image, Model model) {
        mController = controller;
        mNeuralNetwork = network;
        mImage = image;
        mModel = model;

        Set<String> inputNames = mNeuralNetwork.getInputTensorsNames();
        Set<String> outputNames = mNeuralNetwork.getOutputTensorsNames();
        if (inputNames.size() != 1 || outputNames.size() != 1) {
            throw new IllegalStateException("Invalid network input and/or output tensors.");
        } else {
            mInputLayer = inputNames.iterator().next();
            mOutputLayer = outputNames.iterator().next();
        }

    }

    @Override
    protected void onPostExecute(String[] labels) {
        super.onPostExecute(labels);
        if (labels.length > 0) {
            mController.onClassificationResult(labels, mJavaExecuteTime);
        } else {
            mController.onClassificationFailed();
        }
    }

    void loadMeanImageIfAvailable(File meanImage, final int imageSize) {
        ByteBuffer buffer = ByteBuffer.allocate(imageSize * FLOAT_SIZE)
                .order(ByteOrder.nativeOrder());
        if (!meanImage.exists()) {
            return;
        }
        FileInputStream fileInputStream = null;
        try {
            fileInputStream = new FileInputStream(meanImage);
            final byte[] chunk = new byte[1024];
            int read;
            while ((read = fileInputStream.read(chunk)) != -1) {
                buffer.put(chunk, 0, read);
            }
            buffer.flip();
        } catch (IOException e) {
            buffer = ByteBuffer.allocate(imageSize * FLOAT_SIZE);
        } finally {
            if (fileInputStream != null) {
                try {
                    fileInputStream.close();
                } catch (IOException e) {
                    // Do thing
                }
            }
        }
        mMeanImage = buffer.asFloatBuffer();
    }

    float[] loadRgbBitmapAsFloat(Bitmap image) {
        final int[] pixels = new int[image.getWidth() * image.getHeight()];
        image.getPixels(pixels, 0, image.getWidth(), 0, 0,
                image.getWidth(), image.getHeight());

        final float[] pixelsBatched = new float[pixels.length * 3];
        for (int y = 0; y < image.getHeight(); y++) {
            for (int x = 0; x < image.getWidth(); x++) {
                final int idx = y * image.getWidth() + x;
                final int batchIdx = idx * 3;

                final float[] rgb = extractColorChannels(pixels[idx]);
                pixelsBatched[batchIdx]     = rgb[0];
                pixelsBatched[batchIdx + 1] = rgb[1];
                pixelsBatched[batchIdx + 2] = rgb[2];
            }
        }
        return pixelsBatched;
    }

    float[] loadGrayScaleBitmapAsFloat(Bitmap image) {
        final int[] pixels = new int[image.getWidth() * image.getHeight()];
        image.getPixels(pixels, 0, image.getWidth(), 0, 0,
            image.getWidth(), image.getHeight());

        final float[] pixelsBatched = new float[pixels.length];
        for (int y = 0; y < image.getHeight(); y++) {
            for (int x = 0; x < image.getWidth(); x++) {
                final int idx = y * image.getWidth() + x;

                final int rgb = pixels[idx];
                final float b = ((rgb)       & 0xFF);
                final float g = ((rgb >>  8) & 0xFF);
                final float r = ((rgb >> 16) & 0xFF);
                float grayscale = (float) (r * 0.3 + g * 0.59 + b * 0.11);

                pixelsBatched[idx] = preProcess(grayscale);
            }
        }
        return pixelsBatched;
    }

    Pair<Integer, Float>[] topK(int k, final float[] tensor) {
        final boolean[] selected = new boolean[tensor.length];
        final Pair<Integer, Float> topK[] = new Pair[k];
        int count = 0;
        while (count < k) {
            final int index = top(tensor, selected);
            selected[index] = true;
            topK[count] = new Pair<>(index, tensor[index]);
            count++;
        }
        return topK;
    }

    private int top(final float[] array, boolean[] selected) {
        int index = 0;
        float max = -1.f;
        for (int i = 0; i < array.length; i++) {
            if (selected[i]) {
                continue;
            }
            if (array[i] > max) {
                max = array[i];
                index = i;
            }
        }
        return index;
    }

    private float[] extractColorChannels(int pixel) {
        String modelName = mModel.name;

        float b = ((pixel)       & 0xFF);
        float g = ((pixel >>  8) & 0xFF);
        float r = ((pixel >> 16) & 0xFF);

        if (modelName.equals("inception_v3")) {
            return new float[] {preProcess(r), preProcess(g), preProcess(b)};
        } else if (modelName.equals("alexnet") && mMeanImage != null) {
            return new float[] {preProcess(b), preProcess(g), preProcess(r)};
        } else if (modelName.equals("googlenet") && mMeanImage != null) {
            return new float[] {preProcess(b), preProcess(g), preProcess(r)};
        } else {
            return new float[] {preProcess(r), preProcess(g), preProcess(b)};
        }
    }

    private float preProcess(float original) {
        String modelName = mModel.name;

        if (modelName.equals("inception_v3")) {
            return (original - 128) / 128;
        } else if (modelName.equals("alexnet") && mMeanImage != null) {
            return original - mMeanImage.get();
        } else if (modelName.equals("googlenet") && mMeanImage != null) {
            return original - mMeanImage.get();
        } else {
            return original;
        }
    }

    float getMin(float[] array) {
        float min = Float.MAX_VALUE;
        for (float value : array) {
            if (value < min) {
                min = value;
            }
        }
        return min;
    }

    float getMax(float[] array) {
        float max = Float.MIN_VALUE;
        for (float value : array) {
            if (value > max) {
                max = value;
            }
        }
        return max;
    }
}
