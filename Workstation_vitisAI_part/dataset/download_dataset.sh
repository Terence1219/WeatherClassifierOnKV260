#!/bin/bash

# download lane following dataset
echo -e "\e[93m Download lane following dataset ... \e[0m"
gdown --id 1AbVqa2KXUDwX7VuxTvpURR34uKzUbtxo -O Trail_dataset.zip
unzip -o -q Trail_dataset.zip
rm -r Trail_dataset.zip
echo -e "\e[93m Downlaod finished. \e[0m"
