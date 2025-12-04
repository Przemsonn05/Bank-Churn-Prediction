#!/usr/bin/env python
# coding: utf-8

import os
import joblib

def save_plot(fig, filename, path='../images/', dpi=300):
    
    os.makedirs(path, exist_ok=True)
    fig.savefig(os.path.join(path, filename), dpi=dpi, bbox_inches='tight')

def load_model(model_path):
    
    return joblib.load(model_path)

def save_model(model, model_path):
    
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(model, model_path)