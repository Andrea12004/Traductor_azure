
import os
import requests
import logging

logger = logging.getLogger(__name__)

AZURE_BASE_URL = "https://traductormodels.blob.core.windows.net/models"

def download_file(url, local_path):
    """Descarga un archivo desde una URL"""
    try:
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        
        logger.info(f"Descargando {url}...")
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        with open(local_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        logger.info(f"✅ Descargado: {local_path}")
        return True
    except Exception as e:
        logger.error(f"❌ Error descargando {url}: {e}")
        return False

def download_models():
    """Descarga todos los modelos necesarios"""
    models_dir = "models"
    
    files_to_download = [
        ("actions_40.keras", f"{models_dir}/actions_40.keras"),
        ("static_letters_model.keras", f"{models_dir}/static_letters_model.keras"),
        ("words.json", f"{models_dir}/words.json")
    ]
    
    for filename, local_path in files_to_download:
        if not os.path.exists(local_path):
            url = f"{AZURE_BASE_URL}/{filename}"
            success = download_file(url, local_path)
            if not success:
                logger.error(f"No se pudo descargar {filename}")
                return False
        else:
            logger.info(f"✅ Ya existe: {local_path}")
    
    return True