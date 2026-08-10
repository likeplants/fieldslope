# -------------------------
# Create environment
# -------------------------

conda create -n server_env python=3.11 --override-channels -c conda-forge
conda activate server_env


# -------------------------
# Use libmamba solver
# Use only conda-forge
# -------------------------

conda config --env --set solver libmamba
conda config --env --set channel_priority strict


# -------------------------
# Install all dependencies
# -------------------------
conda install --override-channels -c conda-forge fastapi uvicorn numpy pandas requests sqlalchemy pyjwt bcrypt openpyxl python-multipart python-dotenv python-decouple

conda install --override-channels -c conda-forge scipy matplotlib pillow tqdm scikit-image

conda install --override-channels -c conda-forge rasterio gdal libgdal-jp2openjpeg


# -------------------------
# GDAL JP2OpenJPEG plugin path
# -------------------------

conda env config vars set GDAL_DRIVER_PATH="$env:CONDA_PREFIX\Library\lib\gdalplugins"
