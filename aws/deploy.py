import sagemaker
from sagemaker.tensorflow import TensorFlowModel

sagemaker_session = sagemaker.Session()

role = 'arn:aws:iam::863518420170:role/myfirst'

model = TensorFlowModel(
    model_data='s3://khyati-potato-bucket/model.tar.gz',  # Replace with your S3 path
    role=role,
    framework_version='2.14',  # Match this with your TensorFlow version
    sagemaker_session=sagemaker_session
)

predictor = model.deploy(
    initial_instance_count=1,           # Number of instances
    instance_type='ml.m4.xlarge'         # Instance type, can be changed as needed
)


print(f"Model deployed to endpoint: {predictor.endpoint_name}")