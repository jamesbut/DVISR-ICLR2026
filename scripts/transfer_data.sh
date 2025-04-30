#!/bin/bash

# This script packages up data on a server and transfers it to this machine.
# Run script from inside the scripts/ directory

server_path="~/BayesianSymbolicRegression"
server_name="myriad"
data_dir_name="results"
zip_file_name=$server_name"_"$data_dir_name".zip"

# Move all experiment data from Scratch/workspace/ into results/ directory
# Zip results and remove newly created results/workspace
ssh $server_name << EOF
    cp -r ~/Scratch/workspace $server_path/$data_dir_name
    cd $server_path
    zip -r $zip_file_name $data_dir_name
    rm -r $data_dir_name/workspace
EOF

# Transfer data from server
scp $server_name:$server_path"/"$zip_file_name ../$data_dir_name

# Unzip server data and rename
cd ../$data_dir_name
unzip $zip_file_name -d $server_name"_"$data_dir_name

# Delete zip file locally
rm $zip_file_name

# Delete zip file and all files in workspace/ on server
ssh $server_name << EOF
    cd $server_path
    rm $zip_file_name
    rm ~/Scratch/workspace/*
EOF
