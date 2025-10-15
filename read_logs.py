import os
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing import event_accumulator

def read_tensorboard_logs(log_dir, output_dir="log_plots"):
    """
    Reads TensorBoard event files, prints scalar data, and saves plots.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    event_files = [os.path.join(log_dir, f) for f in os.listdir(log_dir) if f.startswith("events.out.tfevents")]

    if not event_files:
        print(f"No TensorBoard event files found in {log_dir}")
        return

    for event_file in event_files:
        print(f"Reading event file: {event_file}")
        ea = event_accumulator.EventAccumulator(event_file,
                                                size_guidance={event_accumulator.SCALARS: 0})
        ea.Reload()

        tags = ea.Tags()['scalars']
        if not tags:
            print("No scalar data found in this event file.")
            continue

        for tag in tags:
            print(f"\n--- Tag: {tag} ---")
            events = ea.Scalars(tag)
            steps = [event.step for event in events]
            values = [event.value for event in events]

            for i in range(len(steps)):
                print(f"Step: {steps[i]}, Value: {values[i]:.4f}")

            plt.figure(figsize=(10, 5))
            plt.plot(steps, values)
            plt.title(tag)
            plt.xlabel("Step")
            plt.ylabel("Value")
            plt.grid(True)
            
            # Sanitize tag name for filename
            safe_tag = tag.replace('/', '_')
            plot_filename = os.path.join(output_dir, f"{safe_tag}.png")
            plt.savefig(plot_filename)
            plt.close()
            print(f"Saved plot to {plot_filename}")
            
        print("-" * (len(tag) + 12))


if __name__ == '__main__':
    log_directory = "logs"
    read_tensorboard_logs(log_directory)