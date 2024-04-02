## SHELL SCRPIT IS FOR WHAT COMMAND WE RUN CMD WE WRITE HERE

echo $(date): "START"


echo $(date): "Creating env with python 3.8 version"

conda create --prefix ./env python=3.8 -y

echo $(date): "activating the conda enverionment"

source activate ./env

echo $(date):"Install the dev requiremnets"

pip install -r requirements_dev.txt

echo $(date):"END"
