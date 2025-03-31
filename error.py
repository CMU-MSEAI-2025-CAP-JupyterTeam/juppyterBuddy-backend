AIMessage(
    content="",
    additional_kwargs={
        "tool_calls": [
            {
                "id": "call_iT5q1mK168PPE2EVpx4RQqUw",
                "function": {
                    "arguments": '{"cell_type":"code","content":"print(\'Hello, World!\')"}',
                    "name": "create_cell",
                },
                "type": "function",
            }
        ],
        "refusal": None,
    },
    response_metadata={
        "token_usage": {
            "completion_tokens": 26,
            "prompt_tokens": 755,
            "total_tokens": 781,
            "completion_tokens_details": {
                "accepted_prediction_tokens": 0,
                "audio_tokens": 0,
                "reasoning_tokens": 0,
                "rejected_prediction_tokens": 0,
            },
            "prompt_tokens_details": {"audio_tokens": 0, "cached_tokens": 0},
        },
        "model_name": "gpt-4-0125-preview",
        "system_fingerprint": None,
        "finish_reason": "tool_calls",
        "logprobs": None,
    },
    id="run-3e53ba96-f5ea-48be-af74-4ae9750455da-0",
    tool_calls=[
        {
            "name": "create_cell",
            "args": {"cell_type": "code", "content": "print('Hello, World!')"},
            "id": "call_iT5q1mK168PPE2EVpx4RQqUw",
            "type": "tool_call",
        }
    ],
    usage_metadata={
        "input_tokens": 755,
        "output_tokens": 26,
        "total_tokens": 781,
        "input_token_details": {"audio": 0, "cache_read": 0},
        "output_token_details": {"audio": 0, "reasoning": 0},
    },
),



