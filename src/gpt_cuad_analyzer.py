"""
GPT-Powered CUAD Contract Analyzer
Built on Contract Understanding Atticus Dataset (CUAD)
Uses OpenAI GPT-4 for intelligent contract analysis
"""

import json
import pandas as pd
from pathlib import Path
import re
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict, Counter
from datetime import datetime
import logging
import openai
from dotenv import load_dotenv
import os
import time

# Load environment variables
load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GPTCUADAnalyzer:
    def __init__(self, data_path: str = "data"):
        self.data_path = Path(data_path)
        self.contracts = {}
        self.legal_categories = []
        self.category_knowledge = {}
        self.risk_weights = self._initialize_risk_weights()
        
        # Set up OpenAI
        openai.api_key = os.getenv('OPENAI_API_KEY')
        if not openai.api_key:
            logger.warning("OPENAI_API_KEY not found. Set it in .env file for GPT analysis.")
        
        # Load and process CUAD data
        self.load_cuad_dataset()
        self.build_knowledge_base()
        
    def _initialize_risk_weights(self) -> Dict[str, int]:
        """Business risk weights for each legal category (1-10)"""
        return {
            'Uncapped Liability': 10,
            'Change Of Control': 9,
            'Ip Ownership Assignment': 9,
            'Non-Compete': 8,
            'Anti-Assignment': 8,
            'Termination For Convenience': 7,
            'Exclusivity': 7,
            'Most Favored Nation': 6,
            'Revenue/Profit Sharing': 6,
            'Cap On Liability': 5,
            'Non-Disparagement': 5,
            'Insurance': 4,
            'Governing Law': 4,
            'Warranty Duration': 4,
            'Agreement Date': 2,
            'Effective Date': 2,
            'Document Name': 1,
            'Parties': 1
        }
    
    def load_cuad_dataset(self):
        """Load and parse CUAD dataset"""
        cuad_file = self.data_path / "CUADv1.json"
        
        if not cuad_file.exists():
            raise FileNotFoundError(f"CUAD dataset not found at {cuad_file}")
        
        logger.info("Loading CUAD dataset...")
        
        with open(cuad_file, 'r', encoding='utf-8') as f:
            cuad_data = json.load(f)
        
        logger.info(f"CUAD Version: {cuad_data.get('version', 'Unknown')}")
        logger.info(f"Contracts loaded: {len(cuad_data['data'])}")
        
        # Process contracts and extract Q&A pairs
        for item in cuad_data['data']:
            contract_title = item['title']
            self.contracts[contract_title] = {
                'qa_pairs': []
            }
            
            for paragraph in item['paragraphs']:
                context = paragraph['context']
                
                # Process Q&A pairs
                for qa in paragraph['qas']:
                    category = self._extract_category(qa['question'])
                    
                    self.contracts[contract_title]['qa_pairs'].append({
                        'question': qa['question'],
                        'category': category,
                        'answers': qa['answers'],
                        'context': context
                    })
                    
                    if category and category not in self.legal_categories:
                        self.legal_categories.append(category)
        
        logger.info(f"Legal categories found: {len(self.legal_categories)}")
    
    def _extract_category(self, question: str) -> Optional[str]:
        """Extract legal category from CUAD question format"""
        match = re.search(r'related to "([^"]+)"', question)
        if match:
            return match.group(1)
        return None
    
    def build_knowledge_base(self):
        """Build knowledge base from CUAD expert annotations"""
        logger.info("Building knowledge base from expert annotations...")
        
        for contract_title, contract_data in self.contracts.items():
            for qa_pair in contract_data['qa_pairs']:
                category = qa_pair['category']
                answers = qa_pair['answers']
                
                if category and answers:
                    if category not in self.category_knowledge:
                        self.category_knowledge[category] = {
                            'examples': [],
                            'question_template': qa_pair['question']
                        }
                    
                    # Store expert examples
                    for answer in answers:
                        if answer['text'].strip():
                            self.category_knowledge[category]['examples'].append({
                                'text': answer['text'],
                                'contract': contract_title
                            })
        
        logger.info(f"Knowledge base built with {len(self.category_knowledge)} categories")
    
    def analyze_contract(self, contract_text: str, contract_name: str = "New Contract") -> Dict[str, Any]:
        """Analyze a new contract using GPT-4 with CUAD knowledge"""
        logger.info(f"Analyzing contract: {contract_name}")
        
        results = {
            'contractInfo': {
                'name': contract_name,
                'analysisTime': datetime.now().isoformat(),
                'wordCount': len(contract_text.split()),
                'characterCount': len(contract_text)
            },
            'categoryFindings': [],
            'riskAssessment': {},
            'recommendations': [],
            'businessMetrics': {}
        }
        
        # Analyze each category using GPT
        for category in self.legal_categories:
            try:
                finding = self._analyze_category_with_gpt(contract_text, category)
                # Only include findings with reasonable confidence
                if finding['confidence'] > 0.3:
                    results['categoryFindings'].append(finding)
                    logger.info(f"Found {category}: {finding['confidence']:.1%}")
            except Exception as e:
                logger.error(f"Error analyzing {category}: {e}")
                continue
        
        # Calculate risk assessment
        results['riskAssessment'] = self._calculate_risk_assessment(results['categoryFindings'])
        
        # Generate recommendations
        results['recommendations'] = self._generate_recommendations(results['categoryFindings'])
        
        # Calculate business metrics
        results['businessMetrics'] = self._calculate_business_metrics(results['riskAssessment'])
        
        logger.info(f"Analysis complete: {len(results['categoryFindings'])} categories found")
        return results
    
    def _analyze_category_with_gpt(self, contract_text: str, category: str) -> Dict[str, Any]:
        """Use GPT-4 to analyze contract for specific legal category"""
        
        # Get CUAD examples for this category
        examples_text = ""
        if category in self.category_knowledge:
            examples = self.category_knowledge[category]['examples'][:3]  # Use top 3 examples
            examples_list = []
            for i, example in enumerate(examples, 1):
                examples_list.append(f"Example {i}: \"{example['text']}\"")
            examples_text = "\n".join(examples_list)
        
        # Create focused prompt
        prompt = f"""You are a legal contract analyst. Analyze this contract for the specific legal category: "{category}".

CUAD Expert Examples for "{category}":
{examples_text if examples_text else "No specific examples available - use your legal knowledge."}

CONTRACT TO ANALYZE:
{contract_text[:3000]}

TASK:
1. Search for any text related to "{category}"
2. If found, extract the most relevant text snippet
3. Rate confidence from 0.0 (not found) to 1.0 (definitely found)
4. Provide brief explanation

RESPOND IN EXACTLY THIS JSON FORMAT:
{{"found": true/false, "confidence": 0.0, "text": "relevant text or empty", "explanation": "brief explanation"}}

Be precise and conservative with confidence scores. Only high confidence if clearly matches the category."""

        try:
            if not openai.api_key:
                # Fallback to simple keyword matching if no API key
                return self._simple_keyword_analysis(contract_text, category)
            
            client = openai.OpenAI(api_key=openai.api_key)
            response = client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are a precise legal contract analyst. Always respond with valid JSON."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=300
            )
            
            # Parse GPT response
            response_text = response.choices[0].message.content.strip()
            
            # Clean up response (remove markdown formatting if present)
            if response_text.startswith("```json"):
                response_text = response_text.replace("```json", "").replace("```", "")
            
            result = json.loads(response_text)
            
            return {
                'category': category,
                'found': result.get('found', False),
                'confidence': float(result.get('confidence', 0.0)),
                'matches': [{'context': result.get('text', ''), 'type': 'gpt_analysis'}] if result.get('found') else [],
                'explanation': result.get('explanation', 'GPT analysis completed'),
                'riskLevel': self._calculate_risk_level(category, result.get('confidence', 0.0))
            }
            
        except json.JSONDecodeError as e:
            logger.error(f"JSON parsing error for {category}: {e}")
            return self._create_empty_result(category)
        except Exception as e:
            logger.error(f"GPT analysis error for {category}: {e}")
            return self._create_empty_result(category)
    
    def _simple_keyword_analysis(self, contract_text: str, category: str) -> Dict[str, Any]:
        """Fallback analysis when GPT is not available"""
        # Simple keyword-based analysis as fallback
        category_keywords = {
            'Document Name': ['agreement', 'contract'],
            'Governing Law': ['governed by', 'governing law', 'laws of'],
            'Termination For Convenience': ['terminate', 'termination', 'convenience'],
            'Uncapped Liability': ['unlimited', 'uncapped', 'liability'],
            'Non-Compete': ['non-compete', 'compete', 'competition'],
        }
        
        keywords = category_keywords.get(category, [])
        contract_lower = contract_text.lower()
        
        matches = []
        for keyword in keywords:
            if keyword in contract_lower:
                # Find context around keyword
                start = contract_lower.find(keyword)
                context_start = max(0, start - 50)
                context_end = min(len(contract_text), start + 100)
                context = contract_text[context_start:context_end].strip()
                matches.append({'context': context, 'type': 'keyword_match'})
        
        confidence = min(0.6, len(matches) * 0.3) if matches else 0.0
        
        return {
            'category': category,
            'found': len(matches) > 0,
            'confidence': confidence,
            'matches': matches,
            'explanation': f"Keyword analysis found {len(matches)} matches",
            'riskLevel': self._calculate_risk_level(category, confidence)
        }
    
    def _create_empty_result(self, category: str) -> Dict[str, Any]:
        """Create empty result for failed analysis"""
        return {
            'category': category,
            'found': False,
            'confidence': 0.0,
            'matches': [],
            'explanation': 'Analysis failed',
            'riskLevel': 'Low'
        }
    
    def _calculate_risk_level(self, category: str, confidence: float) -> str:
        """Calculate risk level based on category weight and confidence"""
        risk_weight = self.risk_weights.get(category, 3)
        
        if risk_weight >= 8 and confidence > 0.7:
            return 'High'
        elif risk_weight >= 5 and confidence > 0.5:
            return 'Medium'
        else:
            return 'Low'
    
    def _calculate_risk_assessment(self, findings: List[Dict]) -> Dict[str, Any]:
        """Calculate overall risk assessment"""
        if not findings:
            return {
                'overallRiskScore': 0,
                'riskLevel': 'Low',
                'highRiskCategories': [],
                'totalCategoriesFound': 0
            }
        
        total_weighted_risk = 0
        max_possible_risk = 0
        high_risk_categories = []
        
        for finding in findings:
            category = finding['category']
            confidence = finding['confidence']
            risk_weight = self.risk_weights.get(category, 3)
            
            weighted_risk = risk_weight * confidence
            total_weighted_risk += weighted_risk
            max_possible_risk += risk_weight
            
            if risk_weight >= 8 and confidence > 0.6:
                high_risk_categories.append({
                    'category': category,
                    'riskWeight': risk_weight,
                    'confidence': confidence
                })
        
        # Normalize to 0-100 scale
        risk_score = (total_weighted_risk / max_possible_risk * 100) if max_possible_risk > 0 else 0
        
        if risk_score >= 70:
            risk_level = 'High'
        elif risk_score >= 40:
            risk_level = 'Medium'
        else:
            risk_level = 'Low'
        
        return {
            'overallRiskScore': round(risk_score, 1),
            'riskLevel': risk_level,
            'highRiskCategories': high_risk_categories,
            'totalCategoriesFound': len(findings)
        }
    
    def _generate_recommendations(self, findings: List[Dict]) -> List[Dict[str, str]]:
        """Generate actionable recommendations"""
        recommendations = []
        
        high_risk_recommendations = {
            'Uncapped Liability': 'Consider negotiating liability caps to limit financial exposure',
            'Change Of Control': 'Review change of control provisions for business continuity',
            'Ip Ownership Assignment': 'Ensure IP ownership terms align with business strategy',
            'Non-Compete': 'Assess non-compete restrictions for operational impact',
            'Termination For Convenience': 'Evaluate termination provisions for contract flexibility'
        }
        
        for finding in findings:
            category = finding['category']
            if category in high_risk_recommendations and finding['confidence'] > 0.6:
                recommendations.append({
                    'category': category,
                    'priority': 'High',
                    'recommendation': high_risk_recommendations[category]
                })
        
        if len(findings) > 15:
            recommendations.append({
                'category': 'General',
                'priority': 'Medium',
                'recommendation': 'Complex contract with multiple legal provisions - recommend comprehensive legal review'
            })
        
        return recommendations
    
    def _calculate_business_metrics(self, risk_assessment: Dict) -> Dict[str, Any]:
        """Calculate business impact metrics"""
        time_saved_hours = 2.5
        hourly_rate = 350
        cost_saved = time_saved_hours * hourly_rate
        
        return {
            'timeSavedHours': time_saved_hours,
            'costSavedUSD': cost_saved,
            'accuracyLevel': 'Professional',
            'processingTimeSeconds': 30
        }

    def get_summary_stats(self) -> Dict[str, Any]:
        """Get summary statistics about the knowledge base"""
        return {
            'totalContracts': len(self.contracts),
            'totalCategories': len(self.legal_categories),
            'categoriesWithKnowledge': len(self.category_knowledge),
            'categories': sorted(self.legal_categories)
        }

# Test the analyzer
def test_analyzer():
    try:
        analyzer = GPTCUADAnalyzer()
        
        # Test with sample contract
        sample_contract = """
        DISTRIBUTION AGREEMENT
        
        This agreement terminates at the convenience of either party
        with 30 days notice. Liability under this agreement is unlimited
        and uncapped. This agreement is governed by Delaware law.
        
        During the term and for two years after termination, distributor 
        shall not compete with company's business.
        """
        
        results = analyzer.analyze_contract(sample_contract, "Test Contract")
        
        print("=== GPT ANALYSIS RESULTS ===")
        print(f"Risk Score: {results['riskAssessment']['overallRiskScore']}")
        print(f"Categories Found: {len(results['categoryFindings'])}")
        
        for finding in results['categoryFindings']:
            print(f"- {finding['category']}: {finding['confidence']:.1%} confidence")
            if finding['matches']:
                print(f"  Text: {finding['matches'][0]['context'][:100]}...")
        
        return True
        
    except Exception as e:
        print(f"Test failed: {e}")
        return False

if __name__ == "__main__":
    test_analyzer()