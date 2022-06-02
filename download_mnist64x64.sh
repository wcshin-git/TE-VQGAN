mkdir -p datasets
cd datasets
wget https://www.dropbox.com/s/eudvvn7vgec1i78/MNIST64x64_Stage1.tar.gz
wget https://www.dropbox.com/s/nzw5qrgtkaypc4y/MNIST64x64_Translated.tar.gz
echo "Extracting..."
tar -zxf MNIST64x64_Stage1.tar.gz
tar -zxf MNIST64x64_Translated.tar.gz
cd ..