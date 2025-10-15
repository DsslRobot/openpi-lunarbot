# Fine-Tuning Guide for OpenPI Models

This guide provides a comprehensive walkthrough for fine-tuning a base OpenPI model (e.g., π₀.₅) on your own custom robot dataset. The process involves three main stages: preparing your data, configuring the training pipeline, and running the training job.

---

## Step 1: Convert Your Data to a `LeRobot` Dataset

The training pipeline requires data to be in the standardized `LeRobot` format. You will create a script to convert your raw data into this format.

1.  **Create a Conversion Script**: Copy `examples/libero/convert_libero_data_to_lerobot.py` to use as a template.

2.  **Define Your Dataset Structure**: In your script, use `LeRobotDataset.create` to define the schema of your data. This is the most critical step.

    ```python
    # In your conversion script
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

    REPO_NAME = "your_hf_username/my_robot_dataset"

    dataset = LeRobotDataset.create(
        repo_id=REPO_NAME,
        robot_type="your_robot_name", # e.g., "ur5"
        fps=10, # Framerate of your recorded data
        features={
            # Add one entry for each camera view
            "image": {
                "dtype": "image",
                "shape": (256, 256, 3),
                "names": ["height", "width", "channel"],
            },
            # This feature for proprioception MUST be named "state"
            "state": {
                "dtype": "float32",
                "shape": (YOUR_STATE_DIM,), # e.g., (8,) for joint angles + gripper
                "names": ["state"],
            },
            # This feature for actions MUST be named "actions"
            "actions": {
                "dtype": "float32",
                "shape": (YOUR_ACTION_DIM,), # e.g., (7,) for a 7-DOF action
                "names": ["actions"],
            },
        },
    )
    ```

3.  **Populate the Dataset**: Loop through your raw data and add it to the dataset frame by frame, episode by episode.

    ```python
    # In your conversion script, replace the existing data loop
    for episode in my_custom_data_loader():
        for step in episode:
            # The dictionary keys must match the features defined above
            dataset.add_frame(
                {
                    "image": step["observation"]["image_view_1"],
                    "state": step["observation"]["robot_state"],
                    "actions": step["action"],
                    # The language instruction is mandatory for VLA models
                    "task": step["language_instruction"],
                }
            )
        # Save the episode to disk after all its frames have been added
        dataset.save_episode()
    ```

4.  **Run the Script**: Execute your script to generate the dataset locally.
    ```bash
    uv run your_conversion_script.py --data_dir /path/to/your/raw/data
    ```

---

## Step 2: Define Training Configurations

Next, you need to configure the training pipeline by creating three components within `src/openpi/training/config.py`.

### 2.1. Create Data Mapping Classes

Create classes that bridge the gap between your dataset's format and the model's internal format. You can add these to a new file (e.g., `src/openpi/policies/my_robot_policy.py`) or directly in `config.py`.

*   **Inputs Class**: Maps data from your dataset to the model's expected input keys.
    ```python
    # In src/openpi/policies/my_robot_policy.py
    import dataclasses
    import numpy as np
    from openpi import transforms
    from openpi.models import model as _model

    @dataclasses.dataclass(frozen=True)
    class MyRobotInputs(transforms.DataTransformFn):
        model_type: _model.ModelType

        def __call__(self, data: dict) -> dict:
            # Map your dataset's image keys to the model's expected keys
            base_image = _parse_image(data["image"]) # Assumes 'image' key from LeRobot dataset

            return {
                "state": data["state"],
                "image": {
                    "base_0_rgb": base_image,
                    # Pad missing camera views with zeros
                    "left_wrist_0_rgb": np.zeros_like(base_image),
                    "right_wrist_0_rgb": np.zeros_like(base_image),
                },
                "image_mask": {
                    "base_0_rgb": np.True_,
                    "left_wrist_0_rgb": np.False_,
                    "right_wrist_0_rgb": np.False_,
                },
                "actions": data.get("actions"), # Pass actions if available (during training)
                "prompt": data.get("prompt"),   # Pass prompt if available
            }
    ```
*   **Outputs Class**: Trims the model's action output to your robot's action dimension.
    ```python
    # In src/openpi/policies/my_robot_policy.py
    @dataclasses.dataclass(frozen=True)
    class MyRobotOutputs(transforms.DataTransformFn):
        def __call__(self, data: dict) -> dict:
            # Replace '7' with your robot's action dimension
            return {"actions": np.asarray(data["actions"][:, :7])}
    ```

### 2.2. Create a `DataConfigFactory`

In `src/openpi/training/config.py`, create a factory that bundles the data processing steps for your dataset.

```python
// filepath: src/openpi/training/config.py
// ...existing code...
// Import your new policy classes
import openpi.policies.my_robot_policy as my_robot_policy

// Add this function to the end of the file
def get_data_config_factory() -> DataConfigFactory:
    return DataConfigFactory(
        # Specify your dataset here
        dataset_name="my_robot_dataset",
        # Register the custom transforms
        input_transform=my_robot_policy.MyRobotInputs,
        output_transform=my_robot_policy.MyRobotOutputs,
    )
```

---

## Step 3: Run the Training Job

With your data prepared and configurations set, you can now run the training job.

1.  **Select a Base Model**: Choose a pre-existing OpenPI model as your base. For instance, π₀.₅.

2.  **Configure the Training Script**: Modify `src/openpi/training/train.py` to use your dataset and configurations.

    ```python
    # In src/openpi/training/train.py
    from openpi.training import Trainer
    from openpi.training.config import get_data_config_factory

    # Get the data config factory for your dataset
    data_config_factory = get_data_config_factory()

    # Initialize the trainer with your configurations
    trainer = Trainer(
        model_name="your_base_model", # e.g., "pi_zero_point_five"
        data_config_factory=data_config_factory,
        # ... other trainer configurations ...
    )

    # Start the training process
    trainer.train()
    ```

3.  **Run the Training Script**: Execute the training script to start fine-tuning the model on your dataset.
    ```bash
    uv run src/openpi/training/train.py
    ```

---

## Additional Notes

*   **Monitoring**: Keep an eye on the training logs to monitor the progress and catch any potential issues early.

*   **Hyperparameter Tuning**: You may need to experiment with different hyperparameters to achieve the best performance for your specific dataset and robot.

*   **Evaluation**: After training, evaluate the model's performance on a separate validation set to ensure it generalizes well to new, unseen data.

*   **Documentation**: Refer to the OpenPI documentation for more detailed information on each component and additional configuration options.

*   **Community and Support**: Engage with the OpenPI community for support, to share your progress, and to learn from others' experiences.

By following this guide, you should be able to successfully fine-tune an OpenPI model on your custom robot dataset, adapting the model to better perform the tasks and maneuvers specific to your robotic application.