import argparse
import os
import sys
import logging
import json
from typing import List, Dict, Any

# Add project root to sys.path (two levels up from src/tools)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.tools.utils import setup_logging
from src.agents.initializer import generate_embeddings

def load_data(file_path: str) -> List[Dict[str, Any]]:
    """Loads data from a JSONL or JSON file."""
    data = []
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
        
    with open(file_path, 'r', encoding='utf-8') as f:
        if file_path.endswith('.jsonl'):
            for line in f:
                if line.strip():
                    data.append(json.loads(line))
        elif file_path.endswith('.json'):
            data = json.load(f)
            if not isinstance(data, list):
                raise ValueError("JSON file must contain a list of objects")
    return data

def main():
    parser = argparse.ArgumentParser(description="Generate NPZ Embedding Index from Data File")
    
    parser.add_argument("--input_file", type=str, required=True, 
                        help="Path to the input data file (.jsonl or .json)")
    parser.add_argument("--output_file", type=str, default=None, 
                        help="Path to the output .npz file (defaults to input_file_name + _idx.npz)")
    parser.add_argument("--embedding_model", type=str, default="text-embedding-3-large", 
                        help="Embedding model to use (default: text-embedding-3-large)")
    parser.add_argument("--text_field", type=str, default="problem", 
                        help="Field name in data to embed (default: 'problem')")
    parser.add_argument("--id_field", type=str, default="item_id", 
                        help="Field name for ID (default: 'item_id'). If missing, uses index.")
    parser.add_argument("--batch_size", type=int, default=100, 
                        help="Batch size for embedding generation")
    parser.add_argument("--max_workers", type=int, default=10, 
                        help="Number of parallel workers")
    
    args = parser.parse_args()
    
    # Setup Logging
    setup_logging()
    logging.info(f"Starting Embedding Generation for {args.input_file}")
    
    try:
        # 1. Load Data
        data = load_data(args.input_file)
        logging.info(f"Loaded {len(data)} items from {args.input_file}")
        
        # 2. Extract Text and IDs
        texts = []
        ids = []
        
        for i, item in enumerate(data):
            # Extract text
            text = item.get(args.text_field)
            if not text:
                logging.warning(f"Item {i} missing text field '{args.text_field}'. Skipping.")
                continue
                
            # Extract ID
            item_id = item.get(args.id_field)  
            if item_id is None and item.get("unique_id"):
                # Fallback to generating an ID or using index
                item_id = f"MATH_{item.get("unique_id").split("/")[-1].split(".json")[0]}"
            else:
                item_id = f"item_{i}"
                
            texts.append(str(text))
            ids.append(str(item_id))
            
        logging.info(f"Prepared {len(texts)} items for embedding.")
        
        # 3. Determine Output Path
        if not args.output_file:
            base_name = os.path.splitext(os.path.basename(args.input_file))[0]
            dir_name = os.path.dirname(args.input_file)
            args.output_file = os.path.join(dir_name, f"{base_name}_{args.embedding_model}_idx.npz")
            
        # Ensure .npz extension
        if not args.output_file.endswith(".npz"):
             args.output_file += ".npz"
             
        logging.info(f"Output will be saved to: {args.output_file}")
        
        # 4. Generate Embeddings
        # Reusing the robust generate_embeddings function from initializer.py
        generate_embeddings(
            input_source=texts,
            embedding_model=args.embedding_model,
            output_path=args.output_file,
            batch_size=args.batch_size,
            max_workers=args.max_workers,
            task_name="Generic",
            ids=ids
        )
        
        logging.info("Done.")
        
    except Exception as e:
        logging.error(f"Failed to generate embeddings: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
