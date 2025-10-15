
##  Q1: Explain the primary differences between TensorFlow and PyTorch. When would you choose one over the other?

1. PyTorch uses a dynamic computation graph ("define-by-run") which is intuitive, Pythonic, and easier to debug. This makes it excellent for research and rapid prototyping.
2. TensorFlow traditionally used static computation graphs requiring an upfront model definition, but since version 2.x it supports eager execution, improving ease of use while retaining static graph benefits for deployment.
3. PyTorch dominates in research and academia, favored for its flexibility and fast experimentation, being used in around 85% of deep learning papers.
4. TensorFlow is stronger in production environments, offering mature tools like TensorFlow Serving, TFLite, and TFX for model deployment and optimization at scale.
5. TensorFlow's static graph enables better optimization and resource management for large-scale models and production deployments, often leading to more efficient GPU utilization and lower memory usage.
6. PyTorch is highly scalable and has improved with features like TorchScript for production, but its dynamic graph can introduce some overhead in large-scale deployments.
7. TensorFlow has a broad, well-established community with extensive industry adoption.
8. PyTorch has a rapidly growing community, especially popular in academic research.

## Q2: Describe two use cases for Jupyter Notebooks in AI development.

1. Exploratory Data Analysis and Visualization
    Jupyter Notebooks provide an interactive platform where AI practitioners can sequentially load, clean, and transform datasets
    while immediately inspecting them through visualizations. This stepwise approach aids in identifying data patterns,
    irregularities, and correlations prior to model development. Libraries such as Matplotlib and Plotly seamlessly integrate to
   generate charts and graphs directly within the notebook, facilitating clear and iterative data exploration.

2. Model Prototyping and Experimentation
    Jupyter Notebooks are well-suited for building and testing machine learning models. The notebook's cell-based structure
   allows developers to execute small segments of code independently, making it easy to tweak data processing steps,
   model design, hyperparameters, and evaluation metrics without running the entire workflow. This leads to faster
   iteration cycles and more efficient refinement before moving models to production.

## Q3: How does spaCy enhance NLP tasks compared to basic Python string operations?

spaCy boosts NLP beyond basic Python string methods by offering smart, 
linguistics-based processing, not just simple text handling. It provides 
key tools like tokenization, part-of-speech tagging, dependency parsing,
and named entity recognition for better understanding of text grammar 
and meaning. spaCy is fast and scales well, using efficient data 
structures and hash-based vocabularies. Its advanced rule-based 
matching beats simple regex or substring searches, letting it 
find complex info like entity links and subtle sentiments that 
basic string operations can’t detect



   
