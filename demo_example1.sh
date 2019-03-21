mkdir images -p && mkdir results -p;
rm images/content1.png -rf;
rm images/style1.png -rf;
rm results/demo_result_example1.png
cd images;
axel -n 1 https://pre00.deviantart.net/f1a6/th/pre/i/2010/019/0/e/country_road_hdr_by_mirre89.jpg --output=content1.png;
axel -n 1 https://nerdist.com/wp-content/uploads/2017/11/Stranger_Things_S2_news_Images_V03-1024x481.jpg --output=style1.png;
convert -resize 50% content1.png content1.png;
convert -resize 50% style1.png style1.png;
cd ..;
python demo.py;
