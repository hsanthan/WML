import ibm_boto3
import json
from ibm_botocore.client import Config, ClientError

# Constants for IBM COS values
COS_ENDPOINT = "https://s3.us-east.cloud-object-storage.appdomain.cloud" # Current list avaiable at https://control.cloud-object-storage.cloud.ibm.com/v2/endpoints
COS_API_KEY_ID = "lh6naU4ptbQ-85U6suVxZqXRDT1tdYIBeu4I1aZk1gqn" # eg "W00YixxxxxxxxxxMB-odB-2ySfTrFBIQQWanc--P3byk"
COS_INSTANCE_CRN = "crn:v1:bluemix:public:cloud-object-storage:global:a/0a752745d5084ac9b390f561ecdd729d:3910469f-f46c-426c-b842-e8e5ddbf216f::" # eg "crn:v1:bluemix:public:cloud-object-storage:global:a/3bf0d9003xxxxxxxxxx1c3e97696b71c:d6f04d83-6c4f-4a62-a165-696756d63903::"

# Create resource
cos_res = ibm_boto3.resource("s3",
    ibm_api_key_id=COS_API_KEY_ID,
    ibm_service_instance_id=COS_INSTANCE_CRN,
    config=Config(signature_version="oauth"),
    endpoint_url=COS_ENDPOINT
)

# Create client
cos_client = ibm_boto3.client("s3",
    ibm_api_key_id=COS_API_KEY_ID,
    ibm_service_instance_id=COS_INSTANCE_CRN,
    config=Config(signature_version="oauth"),
    endpoint_url=COS_ENDPOINT
)


#replicate json and upload to COS
#cos_folder_name = 'STR-Files'
cos_bucket = 'wml-cos'
#filename = 'STR-'

cos_folder_name = 'STR-1'
#filename = 'LVCTR-'

def upload_files_cos(cos_bucket, cos_folder_name, filename):
    num_files = 1000
    with open('LVCTR-1.json', 'r') as str1:
        dest = json.loads(str1.read())
        for i in range(num_files):
            destfile = cos_folder_name+'/'+ filename + str(i) + '.json'
            with open(destfile, 'w', encoding="utf-8") as str1_copy:
                json.dump(dest, str1_copy, indent=2)

                #cos_client.put_object(Bucket='wml-cos',Body='', Key='test-folder/')
                try:
                    res = cos_client.upload_file(destfile, cos_bucket, destfile)
                except Exception as e:
                    print(Exception, e)
                else:
                    print('File Uploaded')

def delete_items(bucket_name, folder_name):
    try:
        delete_request = {
            "Objects": [
                { "Key": folder_name }
            ]
        }

        response = cos_client.delete_objects(
            Bucket=bucket_name,
            Delete=delete_request
        )

        print("Deleted items for {0}\n".format(bucket_name))
        print(json.dumps(response.get("Deleted"), indent=4))
    except ClientError as be:
        print("CLIENT ERROR: {0}\n".format(be))
    except Exception as e:
        print("Unable to copy item: {0}".format(e))


#delete_items(cos_bucket, cos_folder_name)

# Delete item
def delete_item(bucket_name, item_name):
    print("Deleting item: {0} from bucket: {1}".format(item_name, bucket_name))
    try:
        cos_client.delete_object(
            Bucket=bucket_name,
            Key=item_name
        )
        print("Item: {0} deleted!".format(item_name))
        #log_done()
    except ClientError as be:
        log_client_error(be)
    except Exception as e:
        log_error("Unable to delete item: {0}".format(e))

for i in range(1000):
    filename = 'STR-1/STR' + str(i) + '.json'
    delete_item(cos_bucket, filename)