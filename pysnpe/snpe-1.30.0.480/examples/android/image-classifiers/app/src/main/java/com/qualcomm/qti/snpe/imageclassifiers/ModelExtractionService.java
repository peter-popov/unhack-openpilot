/*
 * Copyright (c) 2016 Qualcomm Technologies, Inc.
 * All Rights Reserved.
 * Confidential and Proprietary - Qualcomm Technologies, Inc.
 */
package com.qualcomm.qti.snpe.imageclassifiers;

import android.app.IntentService;
import android.content.Intent;
import android.content.Context;
import android.net.Uri;
import android.util.Log;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.util.zip.ZipEntry;
import java.util.zip.ZipInputStream;

public class ModelExtractionService extends IntentService {

    private static final String LOG_TAG = ModelExtractionService.class.getSimpleName();
    private static final String ACTION_EXTRACT = "extract";
    private static final String EXTRA_MODEL_RAW_RES_ID = "model_raw_res";
    private static final String EXTRA_MODEL_NAME = "model_name";
    public static final String MODELS_ROOT_DIR = "models";
    private static final int CHUNK_SIZE = 1024;

    public ModelExtractionService() {
        super("ModelExtractionService");
    }

    public static void extractModel(final Context context, final String modelName,
        final int modelRawResId) {
        Intent intent = new Intent(context, ModelExtractionService.class);
        intent.setAction(ACTION_EXTRACT);
        intent.putExtra(EXTRA_MODEL_NAME, modelName);
        intent.putExtra(EXTRA_MODEL_RAW_RES_ID, modelRawResId);
        context.startService(intent);
    }

    @Override
    protected void onHandleIntent(Intent intent) {
        if (intent != null) {
            final String action = intent.getAction();
            if (ACTION_EXTRACT.equals(action)) {
                final int modelRawResId = intent.getIntExtra(EXTRA_MODEL_RAW_RES_ID, 0);
                final String modelName = intent.getStringExtra(EXTRA_MODEL_NAME);
                handleModelExtraction(modelName, modelRawResId);
            }
        }
    }

    private void handleModelExtraction(final String modelName, final int modelRawResId) {
        ZipInputStream zipInputStream = null;
        try {
            final File modelsRoot = getOrCreateExternalModelsRootDirectory();
            final File modelRoot = createModelDirectory(modelsRoot, modelName);
            if (modelExists(modelRoot)) {
                return;
            }

            zipInputStream = new ZipInputStream(getResources().openRawResource(modelRawResId));
            ZipEntry zipEntry;
            while ((zipEntry = zipInputStream.getNextEntry()) != null) {
                final File entry = new File(modelRoot, zipEntry.getName());
                if (zipEntry.isDirectory()) {
                    doCreateDirectory(entry);
                } else {
                    doCreateFile(entry, zipInputStream);
                }
                zipInputStream.closeEntry();
            }
            getContentResolver().notifyChange(
                Uri.withAppendedPath(Model.MODELS_URI, modelName), null);
        } catch (IOException e) {
            Log.e(LOG_TAG, e.getMessage(), e);
            try {
                if (zipInputStream != null) {
                    zipInputStream.close();
                }
            } catch (IOException ignored) {}
            getContentResolver().notifyChange(Model.MODELS_URI, null);
        }
    }

    private boolean modelExists(File modelRoot) {
        return modelRoot.listFiles().length > 0;
    }

    private void doCreateFile(File file, ZipInputStream inputStream) throws IOException {
        final FileOutputStream outputStream = new FileOutputStream(file);
        final byte[] chunk = new byte[CHUNK_SIZE];
        int read;
        while ((read = inputStream.read(chunk)) != -1) {
            outputStream.write(chunk, 0, read);
        }
        outputStream.close();
    }

    private void doCreateDirectory(File directory) throws IOException {
        if (!directory.mkdirs()) {
            throw new IOException("Can not create directory: " + directory.getAbsolutePath());
        }
    }

    private File getOrCreateExternalModelsRootDirectory() throws IOException {
        final File modelsRoot = getExternalFilesDir(MODELS_ROOT_DIR);
        if (modelsRoot == null) {
            throw new IOException("Unable to access application external storage.");
        }

        if (!modelsRoot.isDirectory() && !modelsRoot.mkdir()) {
            throw new IOException("Unable to create model root directory: " +
                modelsRoot.getAbsolutePath());
        }
        return modelsRoot;
    }

    private File createModelDirectory(File modelsRoot, String modelName) throws IOException {
        final File modelRoot = new File(modelsRoot, modelName);
        if (!modelRoot.isDirectory() && !modelRoot.mkdir()) {
            throw new IOException("Unable to create model root directory: " +
                modelRoot.getAbsolutePath());
        }
        return modelRoot;
    }

}
