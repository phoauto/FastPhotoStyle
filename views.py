from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework import status
from django.http import HttpResponse

from PIL import Image
import requests
from io import BytesIO

import torch
from . import process_stylization
from .photo_wct import PhotoWCT
from .photo_gif import GIFSmoothing

p_wct = PhotoWCT()
p_wct.load_state_dict(torch.load('./gan/PhotoWCTModels/photo_wct.pth'))
# p_wct.cuda(0)
p_pro = GIFSmoothing(r=35, eps=0.001)

size = 128, 128

@api_view(['GET'])
def image(request):
    try:
        params = request.GET
        style = params.get('style', '')
        img = Image.open('./gan/images/' + style + '.jpg')
        img.thumbnail(size, Image.ANTIALIAS)

        imgBytes = BytesIO()
        img.save(imgBytes, format='jpg')
        return HttpResponse(imgBytes.getvalue(), content_type="image/jpg")
    except ValueError as e:
        return Response(e.args[0],status.HTTP_400_BAD_REQUEST)

@api_view(['GET'])
def style(request):
    try:
        params = request.GET
        url = params.get('url', '')
        cont_img = Image.open(BytesIO(requests.get(url).content)).convert('RGB')

        style = params.get('style', 'sunset1')
        styl_img = Image.open('./gan/images/' + style + '.jpg').convert('RGB')

        img = process_stylization.stylization(
            stylization_module=p_wct,
            smoothing_module=p_pro,
            cont_img=cont_img,
            styl_img=styl_img,
            content_seg_path=[],
            style_seg_path=[],
            cuda=0,
            save_intermediate=False,
            no_post=False
        )
        imgBytes = BytesIO()
        img.save(imgBytes, format='PNG')
        return HttpResponse(imgBytes.getvalue(), content_type="image/png")
    except ValueError as e:
        return Response(e.args[0],status.HTTP_400_BAD_REQUEST)
