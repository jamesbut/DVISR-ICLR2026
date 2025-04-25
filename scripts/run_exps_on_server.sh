#!/bin/bash

# This script transfers run scripts to a server and executes those
# scripts on that server.
# The script and commands in question are different depending on whether
# they are being ran on the myriad system or a custom server.
# This script should be ran from this directory.

server_name="myriad"
server_path="~/BayesianSymbolicRegression"
exp_args_file="exp_config_args.txt"

# TODO: Add case where exp_config_args.txt is empty

# Read config modification args from exp_args_file
exp_config_args=()
while IFS= read -r line; do
    exp_config_args+=("$line")
done < "$exp_args_file"

# Remove any config args that begin with #
exp_config_args=($(for i in "${exp_config_args[@]}"; do [[ $i != \#* ]] && echo "$i"; done))

# If exp_config_args is empty add "" as an element so that at least one experiment is ran with the default configs.
if [ ${#exp_config_args[@]} -eq 0 ]; then
  exp_config_args+=("")
fi

# Create jobscripts folder and copy myriad_jobscript.sh in there to use as
# a template
mkdir -p jobscripts
cp myriad_jobscript.sh jobscripts

script_filenames=()
for ((i = 0; i < ${#exp_config_args[@]}; i++)); do

    # Create new file for each config args
    file_name="jobscripts/myriad_jobscript_$i.sh"
    script_filenames+=("$file_name")
    cp jobscripts/myriad_jobscript.sh $file_name

    # Read last line of file
    last_line=$(tail -n 1 "$file_name")

    # Create new line
    # new_line="$last_line ${exp_config_args[$i]} < outs/out$i.txt"
    new_line="$last_line ${exp_config_args[$i]}"

    # Modify last line in file
    { sed '$d' $file_name; echo $new_line; } > tmp && mv tmp $file_name

done

# Remove template jobscript
rm jobscripts/myriad_jobscript.sh

# Create scripts/jobscripts folder on myriad if it doesn't yet exist
ssh $server_name "mkdir -p $server_path/scripts/jobscripts"

# Transfer all jobscripts to myriad
scp jobscripts/* $server_name:$server_path"/scripts/jobscripts"

# Build command to run all scripts in jobscripts on myriad
qsub_cmd="cd $server_path/scripts && "
for script in "${script_filenames[@]}"; do
  qsub_cmd+="qsub $script && "
done

# Remove all jobscripts on myriad
qsub_cmd+=" rm -r jobscripts/"

echo $qsub_cmd

# Run command on myriad
ssh myriad "$qsub_cmd"

# Remove jobscripts from client
rm -r jobscripts/
