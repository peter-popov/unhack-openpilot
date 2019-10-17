/*
 * Copyright (c) 2016-2018 Qualcomm Technologies, Inc.
 * All Rights Reserved.
 * Confidential and Proprietary - Qualcomm Technologies, Inc.
 */
package com.qualcomm.qti.snpe.imageclassifiers;

import android.app.Application;
import android.graphics.Bitmap;
import android.os.AsyncTask;

import com.qualcomm.qti.snpe.NeuralNetwork;
import com.qualcomm.qti.snpe.SNPE;
import com.qualcomm.qti.snpe.imageclassifiers.tasks.AbstractClassifyImageTask;
import com.qualcomm.qti.snpe.imageclassifiers.tasks.ClassifyImageWithFloatTensorTask;
import com.qualcomm.qti.snpe.imageclassifiers.tasks.ClassifyImageWithUserBufferTf8Task;
import com.qualcomm.qti.snpe.imageclassifiers.tasks.LoadImageTask;
import com.qualcomm.qti.snpe.imageclassifiers.tasks.LoadNetworkTask;

import java.io.File;
import java.lang.ref.SoftReference;
import java.util.Arrays;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.Set;

public class ModelOverviewFragmentController extends AbstractViewController<ModelOverviewFragment> {

    public enum SupportedTensorFormat {
        FLOAT,
        UB_TF8
    }

    private final Map<String, SoftReference<Bitmap>> mBitmapCache;

    private final Model mModel;

    private final Application mApplication;

    private NeuralNetwork mNeuralNetwork;

    private LoadNetworkTask mLoadTask;

    private NeuralNetwork.Runtime mRuntime;

    private SupportedTensorFormat mCurrentSelectedTensorFormat;

    private SupportedTensorFormat mNetworkTensorFormat;

    public ModelOverviewFragmentController(final Application application, Model model) {
        mBitmapCache = new HashMap<>();
        mApplication = application;
        mModel = model;
    }

    @Override
    protected void onViewAttached(ModelOverviewFragment view) {
        view.setModelName(mModel.name);
        view.setSupportedRuntimes(getSupportedRuntimes());
        view.setSupportedTensorFormats(Arrays.asList(SupportedTensorFormat.values()));
        loadImageSamples(view);
    }

    private void loadImageSamples(ModelOverviewFragment view) {
        for (int i = 0; i < mModel.jpgImages.length; i++) {
            final File jpeg = mModel.jpgImages[i];
            final Bitmap cached = getCachedBitmap(jpeg);
            if (cached != null) {
                view.addSampleBitmap(cached);
            } else {
                final LoadImageTask task = new LoadImageTask(this, jpeg);
                task.executeOnExecutor(AsyncTask.THREAD_POOL_EXECUTOR);
            }
        }
    }

    private Bitmap getCachedBitmap(File jpeg) {
        final SoftReference<Bitmap> reference = mBitmapCache.get(jpeg.getAbsolutePath());
        if (reference != null) {
            final Bitmap bitmap = reference.get();
            if (bitmap != null) {
                return bitmap;
            }
        }
        return null;
    }

    private List<NeuralNetwork.Runtime> getSupportedRuntimes() {
        final List<NeuralNetwork.Runtime> result = new LinkedList<>();
        final SNPE.NeuralNetworkBuilder builder = new SNPE.NeuralNetworkBuilder(mApplication);
        for (NeuralNetwork.Runtime runtime : NeuralNetwork.Runtime.values()) {
            if (builder.isRuntimeSupported(runtime)) {
                result.add(runtime);
            }
        }
        return result;
    }

    @Override
    protected void onViewDetached(ModelOverviewFragment view) {
        if (mNeuralNetwork != null) {
            mNeuralNetwork.release();
            mNeuralNetwork = null;
        }
    }

    public void onBitmapLoaded(File imageFile, Bitmap bitmap) {
        mBitmapCache.put(imageFile.getAbsolutePath(), new SoftReference<>(bitmap));
        if (isAttached()) {
            getView().addSampleBitmap(bitmap);
        }
    }

    public void onNetworkLoaded(NeuralNetwork neuralNetwork, final long loadTime) {
        if (isAttached()) {
            mNeuralNetwork = neuralNetwork;
            ModelOverviewFragment view = getView();
            view.setNetworkDimensions(getInputDimensions());
            view.setOutputLayersNames(neuralNetwork.getOutputLayers());
            view.setModelVersion(neuralNetwork.getModelVersion());
            view.setLoadingNetwork(false);
            view.setModelLoadTime(loadTime);
        } else {
            neuralNetwork.release();
        }
        mLoadTask = null;
    }

    public void onNetworkLoadFailed() {
        if (isAttached()) {
            ModelOverviewFragment view = getView();
            view.displayModelLoadFailed();
            view.setLoadingNetwork(false);
        }
        mLoadTask = null;
        mNetworkTensorFormat = null;
    }

    public void classify(final Bitmap bitmap) {
        if (mNeuralNetwork != null) {
            AbstractClassifyImageTask task;
            switch (mNetworkTensorFormat) {
                case UB_TF8:
                    task = new ClassifyImageWithUserBufferTf8Task(this, mNeuralNetwork, bitmap, mModel);
                    break;
                case FLOAT:
                default:
                    task = new ClassifyImageWithFloatTensorTask(this, mNeuralNetwork, bitmap, mModel);
                    break;
            }
            task.executeOnExecutor(AsyncTask.SERIAL_EXECUTOR);

        } else {
            getView().displayModelNotLoaded();
        }
    }

    public void onClassificationResult(String[] labels, long javaExecuteTime) {
        if (isAttached()) {
            ModelOverviewFragment view = getView();
            view.setClassificationResult(labels);
            view.setJavaExecuteStatistics(javaExecuteTime);
        }
    }

    public void onClassificationFailed() {
        if (isAttached()) {
            getView().displayClassificationFailed();
            getView().setJavaExecuteStatistics(-1);
        }
    }

    public void setTargetRuntime(NeuralNetwork.Runtime runtime) {
        mRuntime = runtime;
    }

    public void setTensorFormat(SupportedTensorFormat format) {
        mCurrentSelectedTensorFormat = format;
    }

    public void loadNetwork() {
        if (isAttached()) {
            ModelOverviewFragment view = getView();
            view.setLoadingNetwork(true);
            view.setNetworkDimensions(null);
            view.setOutputLayersNames(new HashSet<String>());
            view.setModelVersion("");
            view.setModelLoadTime(-1);
            view.setJavaExecuteStatistics(-1);
            view.setClassificationHint();

            final NeuralNetwork neuralNetwork = mNeuralNetwork;
            if (neuralNetwork != null) {
                neuralNetwork.release();
                mNeuralNetwork = null;
            }

            if (mLoadTask != null) {
                mLoadTask.cancel(false);
            }

            mNetworkTensorFormat = mCurrentSelectedTensorFormat;
            mLoadTask = new LoadNetworkTask(mApplication, this, mModel, mRuntime, mCurrentSelectedTensorFormat);
            mLoadTask.executeOnExecutor(AsyncTask.SERIAL_EXECUTOR);
        }
    }

    private int[] getInputDimensions() {
        Set<String> inputNames = mNeuralNetwork.getInputTensorsNames();
        Iterator<String> iterator = inputNames.iterator();
        return iterator.hasNext() ? mNeuralNetwork.getInputTensorsShapes().get(iterator.next()) : null;
    }
}
