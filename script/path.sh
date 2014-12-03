
if [ -z "$(echo $PATH | grep $(pwd))" ]; then
    echo "" >> ~/.bashrc
    echo "export PATH=$(pwd)/local/bin:\$PATH" >> ~/.bashrc
    echo "export LD_LIBRARY_PATH=$(pwd)/local/lib:$(pwd)/local/lib64:$(pwd)/local/libexec:\$LD_LIBRARY_PATH" >> ~/.bashrc
    . ~/.bashrc
fi
