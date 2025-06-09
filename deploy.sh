source <(curl -s https://raw.githubusercontent.com/Danar1111/Openpose-Backend/main/install.sh)
cd ~
source <(curl -s https://raw.githubusercontent.com/HADAIZI/TA_Deployment/main/install.sh)
pm2 start run.py --name movenet-backend --interpreter /usr/local/bin/python3.9
cd ~