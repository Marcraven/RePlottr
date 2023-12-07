# Welcome to DonutPlot
![](https://media3.giphy.com/media/3orieZDAp40AhhOOsg/giphy.gif?cid=ecf05e4715zs61rh6n03rpdf1otp7qx9kudsmc9hsm5jxm8j&ep=v1_gifs_search&rid=giphy.gif&ct=g)

### Refreshing Requirements:
`pip install -r requirements.txt`


# Replottr

## Overview

The Replottr is a program designed to extract valuable information from images containing scatter plots. It utilizes YOLOv8 for detecting the position of the dots and OCR (Optical Character Recognition) for extracting information such as x and y ticks, title, and axis labels.

## Features

- **Object Detection:** YOLOv8 is used to identify and locate the position of dots in the scatter plot.

- **Text Recognition:** OCR is employed to extract textual information, including x and y ticks, title, and axis labels.

- **Data Fusion:** The program merges the detected dot positions with the extracted textual information to create a comprehensive dataset.

- **Export Formats:** The extracted data is provided in both JSON and CSV formats for easy integration into various data analysis tools.
