/*
 * Copyright (c) 2016, 2017 Qualcomm Technologies, Inc.
 * All Rights Reserved.
 * Confidential and Proprietary - Qualcomm Technologies, Inc.
 */
package com.qualcomm.qti.snpe.imageclassifiers;

import android.app.Application;
import android.app.Fragment;
import android.content.Context;
import android.os.Bundle;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.AdapterView;
import android.widget.ArrayAdapter;
import android.widget.ListView;
import android.widget.TextView;

import com.qualcomm.qti.snpe.SNPE;

import java.util.Set;

public class ModelCatalogueFragment extends Fragment {

    private ModelCatalogueFragmentController mController;

    private ListView mModelsList;

    private TextView mLoadStatusText;

    public static ModelCatalogueFragment create() {
        return new ModelCatalogueFragment();
    }

    private ModelsAdapter mModelsAdapter;

    @Override
    public View onCreateView(LayoutInflater inflater, ViewGroup container, Bundle savedInstanceState) {
        return inflater.inflate(R.layout.models_list, container, false);
    }

    @Override
    public void onViewCreated(View view, Bundle savedInstanceState) {
        super.onViewCreated(view, savedInstanceState);
        mModelsList = (ListView) view.findViewById(R.id.models_list);
        mLoadStatusText = (TextView) view.findViewById(R.id.models_load_status);
    }

    @Override
    public void onActivityCreated(Bundle savedInstanceState) {
        super.onActivityCreated(savedInstanceState);
        mController = new ModelCatalogueFragmentController(getActivity());

        mModelsAdapter = new ModelsAdapter(getActivity());
        mModelsList.setAdapter(mModelsAdapter);
        mModelsList.setOnItemClickListener(new AdapterView.OnItemClickListener() {
            @Override
            public void onItemClick(AdapterView<?> parent, View view, int position, long id) {
                MainActivity.class.cast(getActivity()).displayModelOverview(
                    mModelsAdapter.getItem(position));
            }
        });
        getActivity().setTitle(getString(R.string.snpe_version,
                SNPE.getRuntimeVersion((Application) getActivity().getApplicationContext())));
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

    public void setExtractingModelMessageVisible(final boolean isVisible) {
        mLoadStatusText.setText(getString(R.string.loading_models));
        mLoadStatusText.setVisibility(isVisible ? View.VISIBLE : View.GONE);
    }

    public void displayModels(Set<Model> models) {
        setExtractingModelMessageVisible(models.isEmpty());
        mModelsAdapter.clear();
        mModelsAdapter.addAll(models);
        mModelsAdapter.notifyDataSetChanged();
    }

    public void showExtractionFailedMessage() {
        mLoadStatusText.setText(R.string.model_extraction_failed);
        mLoadStatusText.setVisibility(View.VISIBLE);
    }

    private static final class ModelsAdapter extends ArrayAdapter<Model> {
        public ModelsAdapter(Context context) {
            super(context, R.layout.models_list_item, R.id.model_name);
        }
    }
}
