# iCAN_for_live_video
該專案引用了https://github.com/vt-vl-lab/iCAN iCAN模型與https://github.com/endernewton/tf-faster-rcnn faster-rcnn模型

iCAN本身是行為偵側算法，輸入的資料內容為faster-rcnn輸出的結果。
因此我們可以簡單得理解iCAN本身並沒有產生bounding box的能力，他只是藉由faster-rcnn的輸出結果中的人與物的相對關係去計算出可能的行為。
詳細iCAN運作原理請看原作者論文https://arxiv.org/pdf/1808.10437v1.pdf

此部分的程式碼專門是屬於結果輸出的部分，因此沒有train部分的修改。大部分都保留原作者的代碼。

weight 的部分請在/ican執行

chmod +x ./misc/download_dataset.sh

./misc/download_dataset.sh 

chmod +x ./misc/download_detection_results.sh 

./misc/download_detection_results.sh

chmod +x ./misc/download_training_data.sh 

./misc/download_training_data.sh

train的程式碼會在另一個project:iCAN_for_train

git clone url之後到frcnn/data/video/資料夾底下放入你想要偵測的video(mp4)
在video.py 內第156行
vc = cv2.VideoCapture("/home/jovyan/faster_rcnn_zip/data/video/mv4.mp4")=> 改成你的檔名路徑
cd 專案資料夾
python video.py 即可運行
在專案資料夾下直接執行live.py即可執行live.py

在運行的過程中應該會有些路徑的bug，再麻煩自行解決。
