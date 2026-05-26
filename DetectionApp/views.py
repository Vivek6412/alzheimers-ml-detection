from django.shortcuts import render
from django.template import RequestContext
from django.contrib import messages
from django.http import HttpResponse
from django.conf import settings
from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.models import User
from django.contrib.auth.decorators import login_required
from .models import UserProfile
import os
import io
import base64
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from . import ml_service

# Constants
UPLOAD_DIR = os.path.join(settings.BASE_DIR, "DetectionApp/static")
DATASET_NAME = "uploaded_dataset.csv"

def index(request):
    if request.method == 'GET':
       return render(request, 'index.html', {})

def UserLogin(request):
    if request.method == 'GET':
        return render(request, 'UserLogin.html', {})

def Register(request):
    if request.method == 'GET':
       return render(request, 'Register.html', {})

def RegisterAction(request):
    if request.method == 'POST':
        username = request.POST.get('t1', False)
        password = request.POST.get('t2', False)
        contact = request.POST.get('t3', False)
        email = request.POST.get('t4', False)
        address = request.POST.get('t5', False)

        if User.objects.filter(username=username).exists():
            output = username + " Username already exists"
        else:
            user = User.objects.create_user(username=username, email=email, password=password)
            UserProfile.objects.create(user=user, contact_no=contact, address=address)
            output = "Signup process completed. Login to perform Alzheimer Disease prediction"

        context = {'data': output}
        return render(request, 'Register.html', context)

def UserLoginAction(request):
    if request.method == 'POST':
        users = request.POST.get('t1', False)
        password = request.POST.get('t2', False)
        user = authenticate(username=users, password=password)
        if user is not None:
            login(request, user)
            request.session['username'] = users
            context = {'data': 'Welcome ' + users}
            return render(request, "UserScreen.html", context)
        else:
            context = {'data': 'Invalid username or password'}
            return render(request, 'UserLogin.html', context)

@login_required(login_url='UserLogin')
def LoadDataset(request):
    if request.method == 'GET':
        return render(request, 'LoadDataset.html', {})

@login_required(login_url='UserLogin')
def LoadDatasetAction(request):    
    if request.method == 'POST':
        # ... logic
        myfile = request.FILES['t1'].read()
        target_path = os.path.join(UPLOAD_DIR, DATASET_NAME)
        
        with open(target_path, "wb") as file:
            file.write(myfile)
        
        dataset = pd.read_csv(target_path)
        columns = dataset.columns
        data = dataset.values
        
        output='<table><thead><tr>'
        for col in columns:
            output += f'<th>{col}</th>'
        output += '</tr></thead><tbody>'
        for i in range(0, min(100, len(data))):
            output += '<tr>'
            for j in range(len(data[i])):
                output += f'<td>{data[i,j]}</td>'
            output += '</tr>'
        output+= "</tbody></table>"
        context= {'data':output}
        return render(request, 'UserScreen.html', context)

@login_required(login_url='UserLogin')
def FeaturesExtraction(request):
    if request.method == 'GET':
        # ... logic
        dataset_path = os.path.join(UPLOAD_DIR, DATASET_NAME)
        if not os.path.exists(dataset_path):
            return render(request, 'UserScreen.html', {'data': 'Please upload dataset first.'})
        
        report = ml_service.process_dataset(dataset_path)
        
        output = f'<div class="status-msg info">Dataset processing completed</div>'
        output += f'<div class="status-msg info">Total records found in dataset = {report["total_records"]}</div>'
        output += f'<div class="status-msg info">Total features before Cognitive Selection = {report["before_pca"]}</div>'
        output += f'<div class="status-msg info">Total features after Cognitive Selection = {report["after_pca"]}</div>'
        
        output += '<div class="mt-6 font-bold text-slate-900">Train & Test Split Details</div>'
        output += f'<div class="status-msg info">80% dataset used to train = {report["train_size"]}</div>'
        output += f'<div class="status-msg info">20% dataset used to test = {report["test_size"]}</div>'
        
        context= {'data':output}
        return render(request, 'UserScreen.html', context)

@login_required(login_url='UserLogin')
def RunML(request):
    if request.method == 'GET':
        try:
            results, img_b64 = ml_service.train_ensemble()
        except Exception as e:
            return render(request, 'UserScreen.html', {'data': str(e)})

        output='<table><thead><tr>'
        columns = ['Algorithm Name', 'Accuracy', 'Precision', 'Recall', 'FSCORE']
        for col in columns:
            output += f'<th>{col}</th>'
        output += '</tr></thead><tbody>'
        
        for name, metrics in results.items():
            output += f'<tr><td>{name}</td>'
            for val in metrics:
                output += f'<td>{val}</td>'
            output += '</tr>'
        output += '</tbody></table>'
        
        context= {'data':output, 'img': img_b64}
        return render(request, 'UserScreen.html', context)

@login_required(login_url='UserLogin')
def Predict(request):
    if request.method == 'GET':
        return render(request, 'Predict.html', {})

@login_required(login_url='UserLogin')
def PredictAction(request):
    if request.method == 'POST':
        # ... logic
        labels = ['No Alzheimer', 'Alzheimer Detected']
        myfile = request.FILES['t1'].read()
        fname = request.FILES['t1'].name
        test_path = os.path.join(UPLOAD_DIR, fname)
        
        with open(test_path, "wb") as file:
            file.write(myfile)
        
        try:
            original_values, predictions = ml_service.predict_data(test_path)
        except Exception as e:
            return render(request, 'UserScreen.html', {'data': str(e)})

        output = '<table><thead><tr><th>Test Data (Raw)</th>'
        output += '<th>Diagnostic Result</th></tr></thead><tbody>'
        
        for i, pred in enumerate(predictions):
            output += f'<tr><td>{original_values[i]}</td>'
            color_class = "success" if pred == 0 else "danger"
            output += f'<td><span class="badge {color_class}">{labels[pred]}</span></td></tr>'
        
        output+= "</tbody></table>"    
        context= {'data':output}
        return render(request, 'UserScreen.html', context)

def UserLogout(request):
    logout(request)
    return render(request, 'index.html', {'data': 'Logged out successfully'})
