import boto3
from decimal import Decimal
from dist_swarm.aws_settings import REGION

dynamodb = boto3.resource('dynamodb', region_name=REGION)

table = dynamodb.Table('unittest')
#get the table keys
tableKeyNames = [key.get("AttributeName") for key in table.key_schema]

#Only retrieve the keys for each item in the table (minimize data transfer)
projectionExpression = ", ".join('#' + key for key in tableKeyNames)
expressionAttrNames = {'#'+key: key for key in tableKeyNames}

counter = 0
page = table.scan(ProjectionExpression=projectionExpression, ExpressionAttributeNames=expressionAttrNames)
with table.batch_writer() as batch:
    while page["Count"] > 0:
        counter += page["Count"]
        # Delete items in batches
        for itemKeys in page["Items"]:
            batch.delete_item(Key=itemKeys)
        # Fetch the next page
        if 'LastEvaluatedKey' in page:
            page = table.scan(
                ProjectionExpression=projectionExpression, ExpressionAttributeNames=expressionAttrNames,
                ExclusiveStartKey=page['LastEvaluatedKey'])
        else:
            break
print(f"Deleted {counter}")