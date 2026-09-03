This model is used to distinguish AI generated art from human art. 
It not only classifies the former but also nullifies it, if needed and have a continuous learning curve to tackle new generations of AI art/image generation
It will also include a dashboard to preview the original, masked and protected images.
A CL evolution/ health tracker will help us learn how far the model has adapted and how much it has learned from the dataset and evolved

📊 How the Continuous Learning Loop Works
Unlike static models that degrade as generative tech evolves, this system employs a Replay Memory Buffer:

Fresh AI art samples are automatically pulled via ingestion scripts.

The system merges new samples with historical weights.

Fine-tuning runs incrementally, ensuring high accuracy across successive generation cycles.

<img width="384" height="480" alt="giphy" src="https://github.com/user-attachments/assets/21dc4251-bd20-48e0-b124-4754e41126e8" />
