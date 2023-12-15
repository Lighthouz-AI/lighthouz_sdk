from marshmallow import fields, Schema


class BenchmarkDetailSchema(Schema):
    class Meta:
        fields = ("query", "expected_response", "context", "put_type", "token_count")

    query = fields.Str(required=True)
    expected_response = fields.Str()
    context = fields.Str()
    put_type = fields.Str()
    token_count = fields.Int()


benchmark_schema = BenchmarkDetailSchema()
