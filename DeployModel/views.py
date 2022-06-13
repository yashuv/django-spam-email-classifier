from django.http import HttpResponse
from django.shortcuts import render

# import our classification model
from . import model


def predict(request):

    userMessage = str(request.POST.get('message'))
    result = model.email_prediction(userMessage)
    return render(request, 'predict.html',  {'result': result})
    

if __name__ == '__main__':
    predict()