/*
 * Copyright (c) 2016-2018 Qualcomm Technologies, Inc.
 * All Rights Reserved.
 * Confidential and Proprietary - Qualcomm Technologies, Inc.
 */
package com.qualcomm.qti.snpe.imageclassifiers;

import android.content.ContentResolver;
import android.content.Context;
import android.database.ContentObserver;
import android.net.Uri;
import android.os.AsyncTask;
import android.os.Handler;

import com.qualcomm.qti.snpe.imageclassifiers.tasks.LoadModelsTask;

import java.util.HashSet;
import java.util.Iterator;
import java.util.Set;

public class ModelCatalogueFragmentController extends
    AbstractViewController<ModelCatalogueFragment> {

    private static final Set<String> mSupportedModels = new HashSet<String>() {{
        add("alexnet");
        add("inception_v3");
        add("googlenet");
    }};

    private final Context mContext;

    ModelCatalogueFragmentController(Context context) {
        mContext = context;
    }

    @Override
    protected void onViewAttached(final ModelCatalogueFragment view) {
        view.setExtractingModelMessageVisible(true);

        final ContentResolver contentResolver = mContext.getContentResolver();
        contentResolver.registerContentObserver(Uri.withAppendedPath(
            Model.MODELS_URI, Model.INVALID_ID), false, mModelExtractionFailedObserver);

        contentResolver.registerContentObserver(Model.MODELS_URI, true, mModelExtractionObserver);

        startModelsExtraction();
        loadModels();
    }

    private void startModelsExtraction() {
        for (Iterator<String> it = mSupportedModels.iterator(); it.hasNext();) {
            String modelName = it.next();
            int resId = getRawResourceId(modelName);
            if (resId == 0) {
                it.remove();
            } else {
                ModelExtractionService.extractModel(mContext, modelName, resId);
            }
        }
    }

    @Override
    protected void onViewDetached(final ModelCatalogueFragment view) {
        final ContentResolver contentResolver = mContext.getContentResolver();
        contentResolver.unregisterContentObserver(mModelExtractionObserver);
        contentResolver.unregisterContentObserver(mModelExtractionFailedObserver);
    }

    private final ContentObserver mModelExtractionObserver =
        new ContentObserver(new Handler()) {
        @Override
        public void onChange(boolean selfChange) {
            super.onChange(selfChange);
            if (isAttached()) {
                loadModels();
            }
        }
    };

    private final ContentObserver mModelExtractionFailedObserver =
        new ContentObserver(new Handler()) {
        @Override
        public void onChange(boolean selfChange) {
            if (isAttached()) {
                getView().showExtractionFailedMessage();
            }
        }
    };

    private void loadModels() {
        final LoadModelsTask task = new LoadModelsTask(mContext, this);
        task.executeOnExecutor(AsyncTask.THREAD_POOL_EXECUTOR);
    }

    public void onModelsLoaded(final Set<Model> models) {
        if (isAttached()) {
            getView().displayModels(models);
        }
    }

    public Set<String> getAvailableModels() {
        return mSupportedModels;
    }

    private int getRawResourceId(String rawName) {
        return mContext.getResources().getIdentifier(rawName, "raw", mContext.getPackageName());
    }

}
