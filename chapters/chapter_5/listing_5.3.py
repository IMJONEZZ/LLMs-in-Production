import json
import grpc
from typing import Optional
from mlserver.codecs.string import StringRequestCodec
import mlserver.grpc.converters as converters
import mlserver.grpc.dataplane_pb2_grpc as dataplane
import mlserver.types as types


def create_request(inputs: dict[str, str]) -> types.InferenceRequest:
    """Setting up the request structure via V2 Inference Protocol"""
    inputs_bytes = json.dumps(inputs).encode("UTF-8")
    return types.InferenceRequest(
        inputs=[
            types.RequestInput(
                name="request",
                shape=[len(inputs_bytes)],
                datatype="BYTES",
                data=[inputs_bytes],
                parameters=types.Parameters(content_type="str"),
            )
        ]
    )


def serialize_request(
    request: types.InferenceRequest,
    model_name: str,
    model_version: Optional[str] = None,
) -> bytes:
    """Serialize the request to Protocol Buffer"""
    return converters.ModelInferRequestConverter.from_types(
        request, model_name=model_name, model_version=model_version
    )


def connect_grpc(host: str) -> dataplane.GRPCInferenceServiceStub:
    """Connect to the gRPC server"""
    grpc_channel = grpc.insecure_channel(host)
    return dataplane.GRPCInferenceServiceStub(grpc_channel)


def make_grpc_request(
    grpc_stub: dataplane.GRPCInferenceServiceStub, serialized_request: bytes
) -> dataplane.ModelInferResponse:
    """Make the gRPC request"""
    return grpc_stub.ModelInfer(serialized_request)


def deserialize_response(
    response: dataplane.ModelInferResponse,
) -> str:
    """Deserialize the response from Protocol Buffer"""
    deserialized_response = converters.ModelInferResponseConverter.to_types(
        response
    )
    return StringRequestCodec.decode_response(deserialized_response)


if __name__ == "__main__":
    # Demonstrate gRPC client
    model_name = "grpc_model"
    inputs = {"message": "I'm using gRPC!"}

    # Make request
    request = create_request(inputs)
    serialized_request = serialize_request(request, model_name)
    grpc_stub = connect_grpc("localhost:8081")
    response = make_grpc_request(grpc_stub, serialized_request)
    print(response)

    # Print deserialized response
    json_text = deserialize_response(response)
    output = json.loads(json_text[0])
    print(output)
