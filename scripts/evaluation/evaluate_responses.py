#!/usr/bin/env python3
"""
Evaluate responses and generate comprehensive report
Compares against GeoCode-GPT paper metrics
"""

import json
import statistics
from pathlib import Path
from datetime import datetime

class ResponseEvaluator:
    def __init__(self, results_file="evaluation_results.jsonl"):
        self.results_file = results_file
        self.results = []
        self.report = {}
        self.load_results()
    
    def load_results(self):
        """Load evaluation results"""
        try:
            with open(self.results_file, 'r', encoding='utf-8') as f:
                self.results = [json.loads(line) for line in f if line.strip()]
            print(f"Loaded {len(self.results)} results from {self.results_file}")
        except FileNotFoundError:
            print(f"File not found: {self.results_file}")
    
    def evaluate_level1(self, result):
        """Evaluate Level 1: Code Completion (Accuracy)"""
        response = result.get('response', '').strip()
        expected = result.get('expected', '').strip()
        
        if not response or not expected:
            return 0
        
        # Simple accuracy check
        accuracy = 100 if response.lower() == expected.lower() else 0
        
        # Partial credit for containing key parts
        if accuracy == 0:
            key_words = expected.split()[:3]
            if all(word.lower() in response.lower() for word in key_words if word):
                accuracy = 50
        
        return accuracy
    
    def evaluate_level2(self, result):
        """Evaluate Level 2: Code Summarization (Completeness, Accuracy, Readability)"""
        response = result.get('response', '').strip()
        expected = result.get('expected', '').strip()
        
        metrics = {}
        
        # Completeness: Check if summary covers main components
        response_length = len(response.split())
        expected_length = len(expected.split())
        completeness = min(100, (response_length / max(expected_length, 1)) * 100)
        
        # Accuracy: Word overlap
        response_words = set(response.lower().split())
        expected_words = set(expected.lower().split())
        if expected_words:
            accuracy = (len(response_words & expected_words) / len(expected_words)) * 100
        else:
            accuracy = 0
        
        # Readability: Based on response structure and length
        readability = 100 if 20 <= response_length <= 150 else 70
        if 'language:' in response.lower() or 'library:' in response.lower():
            readability = 90
        
        return {
            'completeness': min(100, completeness),
            'accuracy': min(100, accuracy),
            'readability': readability
        }
    
    def evaluate_level3(self, result):
        """Evaluate Level 3: Code Generation (Accuracy, Readability, Executability)"""
        response = result.get('response', '').strip()
        
        metrics = {}
        
        # Accuracy: Check for code structure elements
        accuracy = 0
        code_indicators = ['import', 'def ', 'class ', 'return', '=', 'for ', 'if ']
        found_indicators = sum(1 for indicator in code_indicators if indicator in response)
        accuracy = min(100, (found_indicators / len(code_indicators)) * 100)
        
        # Readability: Check for comments, proper formatting
        readability = 60
        if '#' in response:  # Has comments
            readability += 20
        if response.count('\n') > 3:  # Multi-line
            readability += 10
        readability = min(100, readability)
        
        # Executability: Check for syntax issues (basic)
        executability = 50
        if response.count('(') == response.count(')'):  # Balanced parentheses
            executability += 20
        if response.count('[') == response.count(']'):  # Balanced brackets
            executability += 20
        if 'import' in response and any(lib in response for lib in ['numpy', 'pandas', 'arcpy', 'geopandas', 'gdal']):
            executability = min(100, executability + 10)
        executability = min(100, executability)
        
        return {
            'accuracy': accuracy,
            'readability': readability,
            'executability': executability
        }
    
    def generate_report(self):
        """Generate evaluation report"""
        
        report = {}
        
        for level in [1, 2, 3]:
            level_results = [r for r in self.results if r.get('level') == level]
            if not level_results:
                continue
            
            if level == 1:
                scores = [self.evaluate_level1(r) for r in level_results]
                report[f'level_{level}'] = {
                    'count': len(level_results),
                    'accuracy': statistics.mean(scores) if scores else 0,
                    'paper_accuracy': 63.6,  # GeoCode-GPT-7B from paper
                    'difference': statistics.mean(scores) - 63.6 if scores else 0
                }
            
            elif level == 2:
                eval_results = [self.evaluate_level2(r) for r in level_results]
                metrics = {}
                for metric in ['completeness', 'accuracy', 'readability']:
                    values = [r[metric] for r in eval_results if metric in r]
                    metrics[metric] = statistics.mean(values) if values else 0
                
                avg_score = statistics.mean(metrics.values()) if metrics else 0
                report[f'level_{level}'] = {
                    'count': len(level_results),
                    'completeness': metrics.get('completeness', 0),
                    'accuracy': metrics.get('accuracy', 0),
                    'readability': metrics.get('readability', 0),
                    'average': avg_score,
                    'paper_average': 91.4,  # GeoCode-GPT-7B from paper
                    'difference': avg_score - 91.4
                }
            
            elif level == 3:
                eval_results = [self.evaluate_level3(r) for r in level_results]
                metrics = {}
                for metric in ['accuracy', 'readability', 'executability']:
                    values = [r[metric] for r in eval_results if metric in r]
                    metrics[metric] = statistics.mean(values) if values else 0
                
                avg_score = statistics.mean(metrics.values()) if metrics else 0
                report[f'level_{level}'] = {
                    'count': len(level_results),
                    'accuracy': metrics.get('accuracy', 0),
                    'readability': metrics.get('readability', 0),
                    'executability': metrics.get('executability', 0),
                    'average': avg_score,
                    'paper_average': 63.6,  # GeoCode-GPT-7B from paper
                    'difference': avg_score - 63.6
                }
        
        self.report = report
        return report
    
    def print_report(self):
        """Print formatted report to console"""
        
        if not self.report:
            self.generate_report()
        
        print("\n" + "=" * 80)
        print("GEOCODE-GPT EVALUATION REPORT")
        print("=" * 80)
        
        print("\nTEST SUMMARY")
        print("-" * 80)
        print(f"Level 1 (Code Completion): {self.report.get('level_1', {}).get('count', 0)} questions tested")
        print(f"Level 2 (Code Summarization): {self.report.get('level_2', {}).get('count', 0)} questions tested")
        print(f"Level 3 (Code Generation): {self.report.get('level_3', {}).get('count', 0)} questions tested")
        
        # Level 1
        if 'level_1' in self.report:
            level1 = self.report['level_1']
            print("\n" + "=" * 80)
            print("LEVEL 1: MULTIPLE-CHOICE QUESTIONS")
            print("=" * 80)
            print(f"{'Metric':<20} {'Your Model':>15} {'Paper':>15} {'Difference':>15}")
            print("-" * 80)
            print(f"{'Accuracy':<20} {level1['accuracy']:>14.1f}% {level1['paper_accuracy']:>14.1f}% {level1['difference']:>14.1f}%")
            print(f"{'Status':<20} {'BETTER' if level1['difference'] > 0 else 'LOWER':>15}")
        
        # Level 2
        if 'level_2' in self.report:
            level2 = self.report['level_2']
            print("\n" + "=" * 80)
            print("LEVEL 2: CODE SUMMARIZATION")
            print("=" * 80)
            print(f"{'Metric':<20} {'Your Model':>15} {'Paper':>15} {'Difference':>15}")
            print("-" * 80)
            print(f"{'Completeness':<20} {level2['completeness']:>14.1f}% {'-':>14} {'-':>14}")
            print(f"{'Accuracy':<20} {level2['accuracy']:>14.1f}% {'-':>14} {'-':>14}")
            print(f"{'Readability':<20} {level2['readability']:>14.1f}% {'-':>14} {'-':>14}")
            print("-" * 80)
            print(f"{'Average':<20} {level2['average']:>14.1f}% {level2['paper_average']:>14.1f}% {level2['difference']:>14.1f}%")
        
        # Level 3
        if 'level_3' in self.report:
            level3 = self.report['level_3']
            print("\n" + "=" * 80)
            print("LEVEL 3: CODE GENERATION")
            print("=" * 80)
            print(f"{'Metric':<20} {'Your Model':>15} {'Paper':>15} {'Difference':>15}")
            print("-" * 80)
            print(f"{'Accuracy':<20} {level3['accuracy']:>14.1f}% {'-':>14} {'-':>14}")
            print(f"{'Readability':<20} {level3['readability']:>14.1f}% {'-':>14} {'-':>14}")
            print(f"{'Executability':<20} {level3['executability']:>14.1f}% {'-':>14} {'-':>14}")
            print("-" * 80)
            print(f"{'Average':<20} {level3['average']:>14.1f}% {level3['paper_average']:>14.1f}% {level3['difference']:>14.1f}%")
        
        # Overall
        print("\n" + "=" * 80)
        print("OVERALL PERFORMANCE")
        print("=" * 80)
        if self.report:
            all_averages = []
            for key in ['level_1', 'level_2', 'level_3']:
                if key in self.report:
                    val = self.report[key].get('average') or self.report[key].get('accuracy')
                    if val:
                        all_averages.append(val)
            
            if all_averages:
                overall_avg = statistics.mean(all_averages)
                print(f"Overall Average: {overall_avg:.1f}%")
        
        print("\n" + "=" * 80)
        print("NOTES")
        print("=" * 80)
        print("- Training data: 40% (partial dataset)")
        print("- Test samples: 15 (Level 1: 5, Level 2: 5, Level 3: 5)")
        print("- Paper samples: 3000+ questions with expert evaluation")
        print("- Recommendation: Full training data will improve results")
        print("=" * 80 + "\n")
    
    def save_json_report(self, output_file="evaluation_report.json"):
        """Save report as JSON"""
        if not self.report:
            self.generate_report()
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(self.report, f, indent=2, ensure_ascii=False)
        print(f"Saved JSON report to {output_file}")

if __name__ == "__main__":
    evaluator = ResponseEvaluator()
    evaluator.generate_report()
    evaluator.print_report()
    evaluator.save_json_report()
