#!/bin/bash
# Goal: Places each initial BNP structure into a unique directory in a specified path to store the simulation data, while initialising a config.yml file that stores the simulation progress/status for each directory.
# Author: 
# Date:

TARGET_DIR=/scratch/$PROJECT/$USER/SimAnneal 

mkdir $TARGET_DIR
BNPdirs=$(find /g/data/$PROJECT/$USER/InitStruct/BNP -type d)
for dir in $BNPdirs; do
    dirName=${dir:34:${#dir}}
    if [ -z $dirName ]; then continue; fi
    if test ! -d "$TARGET_DIR/$dirName"; then mkdir "$TARGET_DIR/$dirName"; fi
    files=$(find $dir -name "*.lmp")
    for file in $files; do
        tmp=${file:${#dir}:${#file}}
        fileName=${tmp:1:${#tmp}}
        tmp=$((${#fileName} - 4))
        fileDirName=${fileName:0:$tmp}
        if test ! -d "$TARGET_DIR/$dirName/$fileDirName"; then mkdir "$TARGET_DIR/$dirName/$fileDirName"; fi
        if test ! -e "$TARGET_DIR/$dirName/$fileDirName/$fileName"; then
            cp $file "$TARGET_DIR/$dirName/$fileDirName"
            touch $TARGET_DIR/$dirName/$fileDirName/config.yml
        else
            echo "$fileName already exists, skipped!"
            continue
        fi
        echo "Copied $fileName to $TARGET_DIR/$dirName/$fileDirName!"
    done
done
echo "All done!"
