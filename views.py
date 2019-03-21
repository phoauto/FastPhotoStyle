from django.shortcuts import render
from django.http import Http404
from rest_framework.views import APIView
from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework import status
from django.http import HttpResponse
from django.core import serializers
from django.conf import settings

from PIL import Image
import requests
from io import BytesIO

# Create your views here.

@api_view(['GET'])
def style(request):
    try:
        params = request.GET
        url = params.get('url', '')
        print(url)
        content = Image.open(BytesIO(requests.get(url).content))

        style=params.get('style', '1')

        img = content

        imgByteArr = BytesIO()
        img.save(imgByteArr, format='PNG')
        return HttpResponse(imgByteArr.getvalue(), content_type="image/png")
    except ValueError as e:
        return Response(e.args[0],status.HTTP_400_BAD_REQUEST)
