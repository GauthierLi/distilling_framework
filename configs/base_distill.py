models = \
{
    "teacher": {
        "name": "Dinov2Backbone",
        "args": {
            "model_name": 'dinov2_vitg14_reg',
            "intermediate_layers": [9, 19, 29, 39],
        },
        "teacher_out_layers": {
            "indices": [0, 1, 2, 3],
            "shapes":  [

            ]
        }
    },
    "student": {
        "name": "LightClsBackbone",
        "args": {
            "cat_dim": 0,
            "in_channels": 3,
            "widths": [32, 64, 128, 256]
        },
        "student_out_layers": {
            "indices": [0, 1, 2, 3],
            "shapes":  [
                [1, 32, 72, 120],
                [1, 64, 36, 60],
                [1, 128, 18, 30],
                [1, 256, 9, 15]
            ]
        }
    },
}

