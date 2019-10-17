/*
 * Copyright (c) 2016 Qualcomm Technologies, Inc.
 * All Rights Reserved.
 * Confidential and Proprietary - Qualcomm Technologies, Inc.
 */
package com.qualcomm.qti.snpe.imageclassifiers;

public abstract class AbstractViewController<V> {

    private V mView;

    public void attach(V view) {
        if (mView != null) {
            throw new IllegalStateException("A view is already attached.");
        }
        mView = view;
        onViewAttached(view);
    }

    protected abstract void onViewAttached(V view);

    public void detach(V view) {
        if (mView != view) {
            throw new IllegalStateException("Another view is already attached.");
        }
        onViewDetached(view);
        mView = null;
    }

    protected abstract void onViewDetached(V view);

    protected boolean isAttached() {
        return mView != null;
    }

    protected V getView() {
        return mView;
    }
}
