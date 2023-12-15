from marshmallow import fields, Schema


class BenchmarkDetailSchema(Schema):
    class Meta:
        fields = ("query", "expected_response", "context", "put_type")

    query = fields.Str(required=True)
    expected_response = fields.Str()
    context = fields.Str()
    put_type = fields.Str(data_key="type", default="None")


benchmark_schema = BenchmarkDetailSchema()
