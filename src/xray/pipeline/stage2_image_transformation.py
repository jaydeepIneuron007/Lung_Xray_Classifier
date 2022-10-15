from xray.config import ConfigurationManager
from xray.components.image_transformation import TransformData
from xray import logger

STAGE_NAME = "Image Transformation Stage"

def main():
    config = ConfigurationManager()
    transform_data_config = config.get_transform_data_config()
    transformation_data = TransformData(config = transform_data_config)
    transformation_data.run_transformation_data()

if __name__ == '__main__':
    try:
        logger.info(f"*******************")
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e