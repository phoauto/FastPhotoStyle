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

style_names = ("art", "cloud", "fog", "night", "ocean", "ripple", "sea", "sky", "snow", "sunny", "sunset", "water", "winter")
style_nums = (4, 2, 2, 3, 2, 1, 1, 3, 1, 2, 2, 2, 2)
styles = {}
for i, name in enumerate(style_names): 
    for k in range(style_nums[i]):
        key = name + str(k + 1)
        styles[key] = Image.open('./gan/images/' + key + '.jpg').convert('RGB')

size = 128, 128

@api_view(['GET'])
def image(request):
    try:
        params = request.GET
        style = params.get('style', '')
        img = Image.open('./gan/images/' + style + '.jpg')

        width, height = img.size   # Get dimensions
        new_width = width if height / width > 0.75 else height * 4 / 3
        new_height = height if height / width <= 0.75 else width * 3 / 4
        
        left = (width - new_width)/2
        top = (height - new_height)/2
        right = (width + new_width)/2
        bottom = (height + new_height)/2
        img = img.crop((left, top, right, bottom))
        img.thumbnail(size, Image.ANTIALIAS)

        imgBytes = BytesIO()
        img.save(imgBytes, format='png')
        return HttpResponse(imgBytes.getvalue(), content_type="image/png")
    except ValueError as e:
        return Response(e.args[0],status.HTTP_400_BAD_REQUEST)

max_w = 640
max_size = 640, 1024

@api_view(['GET'])
def style(request):
    try:
        params = request.GET
        url = params.get('url', '')
        cont_img = Image.open(BytesIO(requests.get(url).content))
        width, height = cont_img.size
        if width > max_w :
            cont_img.thumbnail(max_size, Image.ANTIALIAS)
        cont_img = cont_img.convert('RGB')

        style_key = params.get('style', 'sunset1')
        styl_img = styles[style_key]

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
