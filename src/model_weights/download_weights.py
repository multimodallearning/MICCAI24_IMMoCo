import os
import wget

def download_weights(url, file_name):
    if not os.path.exists(file_name):
        wget.download(url, file_name)
    else:
        print("File already exists")

def main():
    weights_dir = "src/model_weights"
    file_name = os.path.join(weights_dir, "kLDNet.pth")
    url = "https://cloud.imi.uni-luebeck.de/s/CFpWCNyniFJzwfq/download"
    download_weights(url, file_name)
    
    file_name = os.path.join(weights_dir, "AFPlus.pth")
    url = "https://cloud.imi.uni-luebeck.de/s/TxygJPAJNb6LTjr/download"
    download_weights(url, file_name)
    
    file_name = os.path.join(weights_dir, 
    "classification_model.pth")  
    url = "https://cloud.imi.uni-luebeck.de/s/ky45KRMwdYQmiEY/download"
    download_weights(url, file_name)
    
    file_name = os.path.join(weights_dir,
    "unet_denoising.pth")
    url = "https://cloud.imi.uni-luebeck.de/s/Mnm3993BjisB8d4/download"
    download_weights(url, file_name)
    
    file_name = os.path.join(weights_dir,
    "unet_denoising_classification_task.pth")  
    url = "https://cloud.imi.uni-luebeck.de/s/FAMgjscGyjZHMWo/download"
    download_weights(url, file_name)
    
if __name__ == "__main__":
    main()