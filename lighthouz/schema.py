from marshmallow import Schema, fields


class BenchmarkDetailSchema(Schema):
    class Meta:
        fields = (
            "id",
            "query",
            "expected_response",
            "context",
            "put_type",
            "token_count",
        )

    id = fields.Str(required=True)
    query = fields.Str(required=True)
    expected_response = fields.Str(allow_none=True)
    context = fields.Str(allow_none=True)
    put_type = fields.Str(allow_none=True)
    token_count = fields.Int(allow_none=True)


benchmark_schema = BenchmarkDetailSchema()
