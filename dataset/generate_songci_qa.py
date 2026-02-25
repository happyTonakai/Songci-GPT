#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate multi-turn Q&A conversations from Song Ci data using OpenAI API
"""

import json
import os
import re
import time
from typing import List, Dict, Any
from pathlib import Path
import argparse

try:
    import openai
except ImportError:
    print("Error: openai package is required. Install with: pip install openai")
    exit(1)


class SongCiQAGenerator:
    def __init__(self, api_key: str, model: str = "gpt-3.5-turbo", base_url: str = None, cleanup_partial: bool = True):
        """
        Initialize the Q&A generator

        Args:
            api_key: OpenAI API key
            model: Model to use (default: gpt-3.5-turbo)
            base_url: Optional base URL for custom API endpoint
            cleanup_partial: Whether to clean up partial files after completion (default: True)
        """
        self.api_key = api_key
        self.model = model
        self.base_url = base_url
        self.cleanup_partial = cleanup_partial

        # Note: OpenAI v1.0+ uses client objects, not global configuration

        # Rate limiting
        self.request_delay = 1.0  # seconds between requests

        # Statistics
        self.stats = {
            'processed': 0,
            'success': 0,
            'failed': 0,
            'errors': []
        }

    def create_prompt(self, ci_data: Dict[str, Any]) -> str:
        """
        Create a prompt for generating multi-turn Q&A conversations
        
        Args:
            ci_data: Song Ci data containing author, paragraphs, rhythmic
            
        Returns:
            Formatted prompt string
        """
        author = ci_data.get('author', 'Unknown')
        rhythmic = ci_data.get('rhythmic', '')
        paragraphs = ci_data.get('paragraphs', [])
        
        # Join paragraphs with newlines
        content = '\n'.join(paragraphs)
        
        prompt = f"""你是一位精通中国古典文学的诗词专家。请根据以下宋词内容，创作一个自然流畅的多轮对话问答对。

宋词信息：
- 作者：{author}
- 词牌名：{rhythmic}
- 内容：
{content}

请按照以下要求创作：
1. 生成一个3-5轮的对话问答对
2. 对话要自然流畅，模拟真实的学习或讨论场景
3. 每个回答不超过100个字
4. 问题要围绕词的内容、意境、情感、写作技巧等方面
5. 回答要准确、简洁、有深度
6. 对话可以包含对词句的解读、意境的分析、情感的体会等

请以JSON格式输出，包含以下字段：
- "conversation": 对话列表，每个元素包含"question"和"answer"
- "metadata": 包含作者、词牌名、原始内容

示例输出格式：
{{
  "conversation": [
    {{"question": "问题1", "answer": "回答1"}},
    {{"question": "问题2", "answer": "回答2"}},
    ...
  ],
  "metadata": {{
    "author": "{author}",
    "rhythmic": "{rhythmic}",
    "original_content": "{content[:200]}..."
  }}
}}

请确保回答简洁，每个回答不超过100个字。"""
        
        return prompt

    def generate_qa_pair(self, ci_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate Q&A pair for a single Song Ci
        
        Args:
            ci_data: Song Ci data
            
        Returns:
            Generated Q&A data or None if failed
        """
        try:
            prompt = self.create_prompt(ci_data)
            
            # Call OpenAI API (v1.0+ compatible)
            # Directly in initialization, add custom headers
            client = openai.OpenAI(
                api_key=self.api_key,
                base_url=self.base_url,
                default_headers={
                    "X-Model-Provider-Id": "xiaomi"
                }
            )
            response = client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "你是一位精通中国古典文学的诗词专家，擅长根据宋词内容创作自然流畅的对话问答。"},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=2000,
                stream=False
            )

            # Parse the response
            content = response.choices[0].message.content
            
            # Try to extract JSON from the response
            try:
                # Clean up the response to extract JSON
                json_match = re.search(r'\{[\s\S]*\}', content)
                if json_match:
                    qa_data = json.loads(json_match.group())
                else:
                    # If no JSON found, create a simple structure
                    qa_data = {
                        "conversation": [{"question": "请解读这首词的意境", "answer": content[:100]}],
                        "metadata": {
                            "author": ci_data.get('author', 'Unknown'),
                            "rhythmic": ci_data.get('rhythmic', ''),
                            "original_content": '\n'.join(ci_data.get('paragraphs', []))[:200]
                        }
                    }
                
                # Add original data to metadata
                qa_data['metadata']['original_author'] = ci_data.get('author')
                qa_data['metadata']['original_rhythmic'] = ci_data.get('rhythmic')
                qa_data['metadata']['original_paragraphs'] = ci_data.get('paragraphs', [])
                
                return qa_data
                
            except json.JSONDecodeError as e:
                # If JSON parsing fails, create a fallback structure
                return {
                    "conversation": [
                        {
                            "question": "请解读这首词的意境和情感",
                            "answer": content[:100] if content else "无法生成有效回答"
                        }
                    ],
                    "metadata": {
                        "author": ci_data.get('author', 'Unknown'),
                        "rhythmic": ci_data.get('rhythmic', ''),
                        "original_content": '\n'.join(ci_data.get('paragraphs', []))[:200],
                        "error": f"JSON parsing failed: {str(e)}"
                    }
                }
                
        except Exception as e:
            error_msg = f"Error generating QA for {ci_data.get('author', 'Unknown')}: {str(e)}"
            self.stats['errors'].append(error_msg)
            print(f"\n  [ERROR: {error_msg}]", end="", flush=True)
            return None

    def process_json_file(self, file_path: str, output_dir: str, max_items: int = None, save_interval: int = 10):
        """
        Process a single JSON file containing Song Ci data

        Args:
            file_path: Path to the JSON file
            output_dir: Directory to save results
            max_items: Maximum number of items to process (None for all)
            save_interval: Save results every N items (default: 10)
        """
        print(f"Processing file: {file_path}")

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            if not isinstance(data, list):
                print(f"Warning: {file_path} does not contain a list. Skipping.")
                return

            # Limit items if specified
            if max_items:
                data = data[:max_items]

            total_items = len(data)
            results = []
            last_save_count = 0

            # Create progress bar
            print(f"  Progress: [0/{total_items}] 0%")

            for i, ci_data in enumerate(data):
                self.stats['processed'] += 1

                # Update progress
                progress = (i + 1) / total_items * 100
                print(f"\r  Progress: [{i+1}/{total_items}] {progress:.1f}% - {ci_data.get('author', 'Unknown')} - {ci_data.get('rhythmic', '')}", end="", flush=True)

                qa_result = self.generate_qa_pair(ci_data)

                if qa_result:
                    results.append(qa_result)
                    self.stats['success'] += 1
                else:
                    self.stats['failed'] += 1

                # Save periodically - save immediately after first result, then at intervals
                if len(results) > 0:
                    should_save = (last_save_count == 0) or (len(results) % save_interval == 0) or (len(results) - last_save_count >= save_interval)
                    if should_save:
                        self._save_partial_results(results, output_dir, file_path, i + 1)
                        last_save_count = len(results)

                # Rate limiting
                if i < len(data) - 1:
                    time.sleep(self.request_delay)

            # Final save
            if results:
                output_file = os.path.join(output_dir, f"qa_{os.path.basename(file_path)}")
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(results, f, ensure_ascii=False, indent=2)
                print(f"\n  Completed! Saved {len(results)} Q&A pairs to {output_file}")

                # Clean up partial files
                if self.cleanup_partial:
                    self._cleanup_partial_files(output_dir, file_path)
            else:
                print(f"\n  No results generated for {file_path}")

        except Exception as e:
            print(f"\nError processing {file_path}: {str(e)}")
            self.stats['errors'].append(f"{file_path}: {str(e)}")

    def _save_partial_results(self, results: list, output_dir: str, file_path: str, current_count: int):
        """Save partial results to a temporary file"""
        temp_file = os.path.join(output_dir, f"qa_{os.path.basename(file_path)}.partial.{current_count}")
        try:
            with open(temp_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            print(f"\n  [Auto-saved {len(results)} items to {temp_file}]")
        except Exception as e:
            print(f"\n  [ERROR: Failed to save to {temp_file}: {str(e)}]")

    def _cleanup_partial_files(self, output_dir: str, file_path: str):
        """Clean up partial files after successful completion"""
        base_name = os.path.basename(file_path)
        pattern = f"qa_{base_name}.partial.*"

        import glob
        partial_files = glob.glob(os.path.join(output_dir, pattern))

        if partial_files:
            cleaned_count = 0
            for partial_file in partial_files:
                try:
                    os.remove(partial_file)
                    cleaned_count += 1
                except Exception as e:
                    print(f"\n  [WARNING: Failed to remove {partial_file}: {str(e)}]")

            if cleaned_count > 0:
                print(f"\n  [Cleaned up {cleaned_count} partial file(s)]")
        else:
            print(f"\n  [No partial files to clean up for {base_name}]")

    def process_directory(self, input_dir: str, output_dir: str, max_files: int = None, max_items_per_file: int = None, save_interval: int = 10):
        """
        Process all JSON files in a directory

        Args:
            input_dir: Directory containing Song Ci JSON files
            output_dir: Directory to save results
            max_files: Maximum number of files to process (None for all)
            max_items_per_file: Maximum items per file (None for all)
            save_interval: Save results every N items (default: 10)
        """
        # Create output directory
        try:
            os.makedirs(output_dir, exist_ok=True)
            # Test write permission
            test_file = os.path.join(output_dir, ".write_test")
            with open(test_file, 'w') as f:
                f.write("test")
            os.remove(test_file)
            print(f"Output directory is writable: {os.path.abspath(output_dir)}")
        except Exception as e:
            print(f"ERROR: Cannot write to output directory '{output_dir}': {str(e)}")
            exit(1)

        # Find all JSON files
        json_files = []
        for file in os.listdir(input_dir):
            if file.endswith('.json') and file.startswith('ci.song.'):
                json_files.append(os.path.join(input_dir, file))

        json_files.sort()

        if max_files:
            json_files = json_files[:max_files]

        print(f"Found {len(json_files)} JSON files to process")
        print(f"Auto-save interval: every {save_interval} items")
        print(f"Output directory: {os.path.abspath(output_dir)}")
        print("=" * 60)

        for file_path in json_files:
            self.process_json_file(file_path, output_dir, max_items_per_file, save_interval)
            print()  # Add blank line between files

        # Print statistics
        self.print_stats()

    def print_stats(self):
        """Print processing statistics"""
        print("\n" + "="*50)
        print("PROCESSING STATISTICS")
        print("="*50)
        print(f"Total items processed: {self.stats['processed']}")
        print(f"Successfully generated: {self.stats['success']}")
        print(f"Failed: {self.stats['failed']}")
        print(f"Success rate: {self.stats['success']/max(self.stats['processed'], 1)*100:.1f}%")
        
        if self.stats['errors']:
            print(f"\nErrors encountered: {len(self.stats['errors'])}")
            for error in self.stats['errors'][:5]:  # Show first 5 errors
                print(f"  - {error}")
            if len(self.stats['errors']) > 5:
                print(f"  ... and {len(self.stats['errors']) - 5} more errors")


def main():
    parser = argparse.ArgumentParser(description='Generate Q&A pairs from Song Ci data using OpenAI API')

    parser.add_argument('--input-dir', '-i', default='.',
                       help='Directory containing Song Ci JSON files (default: current directory)')
    parser.add_argument('--output-dir', '-o', default='songci_qa',
                       help='Directory to save generated Q&A files (default: songci_qa)')
    parser.add_argument('--max-files', type=int,
                       help='Maximum number of files to process')
    parser.add_argument('--max-items', type=int,
                       help='Maximum number of items per file to process')
    parser.add_argument('--delay', type=float, default=1.0,
                       help='Delay between API requests in seconds (default: 1.0)')
    parser.add_argument('--save-interval', type=int, default=10,
                       help='Save results every N items (default: 10)')
    parser.add_argument('--no-cleanup', action='store_true',
                       help='Keep partial files after completion (default: False)')

    args = parser.parse_args()

    # Get configuration from environment variables
    api_key = os.getenv('OPENAI_API_KEY')
    model = os.getenv('OPENAI_MODEL', 'gpt-3.5-turbo')
    base_url = os.getenv('OPENAI_BASE_URL')

    # Validate required environment variables
    if not api_key:
        print("Error: OPENAI_API_KEY environment variable is required")
        print("Please set it using: export OPENAI_API_KEY='your-api-key'")
        exit(1)

    # Validate input directory
    if not os.path.isdir(args.input_dir):
        print(f"Error: Input directory '{args.input_dir}' does not exist")
        exit(1)

    # Initialize generator
    generator = SongCiQAGenerator(
        api_key=api_key,
        model=model,
        base_url=base_url,
        cleanup_partial=not args.no_cleanup
    )

    # Set request delay
    generator.request_delay = args.delay

    # Process files
    generator.process_directory(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        max_files=args.max_files,
        max_items_per_file=args.max_items,
        save_interval=args.save_interval
    )


if __name__ == "__main__":
    main()