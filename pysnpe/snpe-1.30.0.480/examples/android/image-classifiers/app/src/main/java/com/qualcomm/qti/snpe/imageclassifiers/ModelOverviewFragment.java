/*
 * Copyright (c) 2016-2018 Qualcomm Technologies, Inc.
 * All Rights Reserved.
 * Confidential and Proprietary - Qualcomm Technologies, Inc.
 */
package com.qualcomm.qti.snpe.imageclassifiers;

import android.app.Application;
import android.app.Fragment;
import android.content.Context;
import android.graphics.Bitmap;
import android.os.Bundle;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.AdapterView;
import android.widget.ArrayAdapter;
import android.widget.Button;
import android.widget.GridView;
import android.widget.ImageView;
import android.widget.Spinner;
import android.widget.TextView;
import android.widget.Toast;

import com.qualcomm.qti.snpe.NeuralNetwork;

import java.util.Arrays;
import java.util.LinkedList;
import java.util.List;
import java.util.Locale;
import java.util.Set;

import com.qualcomm.qti.snpe.imageclassifiers.ModelOverviewFragmentController.SupportedTensorFormat;

public class ModelOverviewFragment extends Fragment {

    public static final String EXTRA_MODEL = "model";

    private static final Locale LOCALE = Locale.CANADA;

    private ModelImagesAdapter mImageGridAdapter;

    private ModelOverviewFragmentController mController;

    private TextView mDimensionsText;

    private TextView mModelNameText;

    private Spinner mOutputLayersSpinners;

    private Spinner mRuntimeSpinner;

    private Spinner mTensorFormatSpinner;

    private TextView mClassificationText;

    private TextView mModelVersionText;

    private TextView mStatisticLoadText;

    private TextView mStatisticJavaExecuteText;

    private Button mBuildButton;

    public static ModelOverviewFragment create(final Model model) {
        final ModelOverviewFragment fragment = new ModelOverviewFragment();
        final Bundle arguments = new Bundle();
        arguments.putParcelable(EXTRA_MODEL, model);
        fragment.setArguments(arguments);
        return fragment;
    }

    @Override
    public View onCreateView(LayoutInflater inflater, ViewGroup container, Bundle savedInstanceState) {
        return inflater.inflate(R.layout.fragment_model, container, false);
    }

    @Override
    public void onViewCreated(View view, Bundle savedInstanceState) {
        super.onViewCreated(view, savedInstanceState);
        GridView imageGrid = (GridView) view.findViewById(R.id.model_image_grid);
        mImageGridAdapter = new ModelImagesAdapter(getActivity());
        imageGrid.setAdapter(mImageGridAdapter);
        imageGrid.setOnItemClickListener(new AdapterView.OnItemClickListener() {
            @Override
            public void onItemClick(AdapterView<?> parent, View view, int position, long id) {
                final Bitmap bitmap = mImageGridAdapter.getItem(position);
                mController.classify(bitmap);
            }
        });

        mModelNameText = (TextView) view.findViewById(R.id.model_overview_name_text);
        mModelVersionText = (TextView) view.findViewById(R.id.model_overview_version_text);
        mDimensionsText = (TextView) view.findViewById(R.id.model_overview_dimensions_text);
        mRuntimeSpinner = (Spinner) view.findViewById(R.id.model_builder_runtime_spinner);
        mTensorFormatSpinner = (Spinner) view.findViewById(R.id.model_builder_tensor_spinner);
        mOutputLayersSpinners = (Spinner) view.findViewById(R.id.model_overview_layers_spinner);
        mClassificationText = (TextView) view.findViewById(R.id.model_overview_classification_text);
        mStatisticLoadText = (TextView) view.findViewById(R.id.model_statistics_init_text);
        mStatisticJavaExecuteText = (TextView) view.findViewById(R.id.model_statistics_java_execute_text);

        mBuildButton = (Button) view.findViewById(R.id.model_build_button);
        mBuildButton.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                mController.loadNetwork();
            }
        });
    }

    @Override
    public void onActivityCreated(Bundle savedInstanceState) {
        super.onActivityCreated(savedInstanceState);
        setHasOptionsMenu(true);
        final Model model = getArguments().getParcelable(EXTRA_MODEL);
        mController = new ModelOverviewFragmentController(
                (Application) getActivity().getApplicationContext(), model);
    }

    @Override
    public void onStart() {
        super.onStart();
        mController.attach(this);
    }

    @Override
    public void onStop() {
        mController.detach(this);
        super.onStop();
    }

    public void addSampleBitmap(Bitmap bitmap) {
        if (mImageGridAdapter.getPosition(bitmap) == -1) {
            mImageGridAdapter.add(bitmap);
            mImageGridAdapter.notifyDataSetChanged();
        }
    }

    public void setNetworkDimensions(int[] inputDimensions) {
        mDimensionsText.setText(inputDimensions != null ? Arrays.toString(inputDimensions) : "");
    }

    public void displayModelLoadFailed() {
        Toast.makeText(getActivity(), R.string.model_load_failed, Toast.LENGTH_SHORT).show();
    }

    public void setModelName(String modelName) {
        mModelNameText.setText(modelName);
    }

    public void setModelVersion(String version) {
        mModelVersionText.setText(version);
    }

    public void setModelLoadTime(long loadTime) {
        if (loadTime > 0) {
            mStatisticLoadText.setText(String.format(LOCALE, "%d ms", loadTime));
        } else {
            mStatisticLoadText.setText(R.string.not_available);
        }
    }

    public void setJavaExecuteStatistics(long javaExecuteTime) {
        if (javaExecuteTime > 0) {
            mStatisticJavaExecuteText.setText(String.format(LOCALE, "%d ms", javaExecuteTime));
        } else {
            mStatisticJavaExecuteText.setText(R.string.not_available);
        }
    }

    public void setOutputLayersNames(Set<String> outputLayersNames) {
        mOutputLayersSpinners.setAdapter(new ArrayAdapter<>(
            getActivity(), android.R.layout.simple_list_item_1,
            new LinkedList<>(outputLayersNames)));
    }

    public void setSupportedTensorFormats(List<SupportedTensorFormat> tensorsFormats) {
        mTensorFormatSpinner.setAdapter(new ArrayAdapter<>(
                getActivity(), android.R.layout.simple_list_item_1, tensorsFormats
        ));

        mTensorFormatSpinner.setOnItemSelectedListener(new AdapterView.OnItemSelectedListener() {
            @Override
            public void onItemSelected(AdapterView<?> adapterView, View view, int i, long l) {
                SupportedTensorFormat format = (SupportedTensorFormat) adapterView.getItemAtPosition(i);
                mController.setTensorFormat(format);
            }

            @Override
            public void onNothingSelected(AdapterView<?> adapterView) {
                mController.setTensorFormat(SupportedTensorFormat.FLOAT);
            }
        });
    }

    public void setSupportedRuntimes(List<NeuralNetwork.Runtime> runtimes) {
        mRuntimeSpinner.setAdapter(new ArrayAdapter<>(
                getActivity(), android.R.layout.simple_list_item_1, runtimes
        ));

        mRuntimeSpinner.setOnItemSelectedListener(new AdapterView.OnItemSelectedListener() {
            @Override
            public void onItemSelected(AdapterView<?> parentView, View selectedItemView, int position, long id) {
                NeuralNetwork.Runtime runtime = (NeuralNetwork.Runtime) parentView.getItemAtPosition(position);
                mController.setTargetRuntime(runtime);
            }

            @Override
            public void onNothingSelected(AdapterView<?> parentView) {
                mController.setTargetRuntime(NeuralNetwork.Runtime.CPU);
            }
        });
    }

    public void setClassificationResult(String[] classificationResult) {
        if (classificationResult.length > 0) {
            mClassificationText.setText(
                    String.format("%s: %s", classificationResult[0], classificationResult[1]));
        } else {
            setClassificationHint();
        }
    }

    public void setClassificationHint() {
        mClassificationText.setText(R.string.classification_hint);
    }

    public void setLoadingNetwork(boolean loading) {
        if (loading) {
            mBuildButton.setText(R.string.loading_network);
            mBuildButton.setEnabled(false);
        } else {
            mBuildButton.setText(R.string.build_network);
            mBuildButton.setEnabled(true);
        }
    }

    public void displayModelNotLoaded() {
        Toast.makeText(getActivity(), R.string.model_not_loaded, Toast.LENGTH_SHORT).show();
    }

    public void displayClassificationFailed() {
        setClassificationHint();
        Toast.makeText(getActivity(), R.string.classification_failed, Toast.LENGTH_SHORT).show();
    }

    private static class ModelImagesAdapter extends ArrayAdapter<Bitmap> {

        ModelImagesAdapter(Context context) {
            super(context, R.layout.model_image_layout);
        }

        @Override
        public View getView(int position, View convertView, ViewGroup parent) {
            View view;
            if (convertView == null) {
                view = LayoutInflater.from(parent.getContext()).
                    inflate(R.layout.model_image_layout, parent, false);
            } else {
                view = convertView;
            }

            final ImageView imageView = ImageView.class.cast(view);
            imageView.setImageBitmap(getItem(position));
            return view;
        }
    }
}
