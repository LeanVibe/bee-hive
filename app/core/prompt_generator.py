"""
LLM-powered Prompt Generator for automated prompt creation and refinement.

Uses meta-prompting techniques to generate optimized prompts based on
task descriptions, performance goals, and domain requirements.
"""

import asyncio
import json
import time
from typing import List, Dict, Any, Optional, Tuple
import structlog
from sqlalchemy.ext.asyncio import AsyncSession

logger = structlog.get_logger()


class PromptGenerator:
    """
    Generates and refines prompts using LLM-powered meta-prompting techniques.
    
    Capabilities:
    - Task-specific prompt generation
    - Domain adaptation and specialization
    - Performance-goal oriented optimization
    - Few-shot learning integration
    - Iterative refinement based on feedback
    """
    
    def __init__(self, db_session: AsyncSession):
        self.db = db_session
        self.logger = logger.bind(component="prompt_generator")
        
        # Configuration
        self.max_prompt_length = 2000
        self.generation_timeout_seconds = 30
        self.template_variations = 5
        
        # Meta-prompting templates
        self.meta_prompt_templates = {
            'task_analysis': """
                Analyze the following task and generate an optimized prompt template:
                
                Task Description: {task_description}
                Domain: {domain}
                Performance Goals: {performance_goals}
                
                Requirements:
                1. Create a clear, specific prompt that guides the AI to perform the task effectively
                2. Include relevant context and examples if beneficial
                3. Structure the prompt for optimal performance on: {performance_goals}
                4. Adapt language and complexity for the {domain} domain
                5. Keep the prompt concise but comprehensive
                
                Generate 3 different prompt variations with explanations for each approach.
            """,
            
            'refinement': """
                Improve the following prompt based on performance feedback:
                
                Current Prompt: {current_prompt}
                Performance Issues: {issues}
                Target Improvements: {improvements}
                Domain: {domain}
                
                Requirements:
                1. Address the specific performance issues identified
                2. Maintain the core intent and functionality
                3. Optimize for {improvements}
                4. Ensure domain-appropriate language and context
                
                Provide an improved version with explanation of changes made.
            """,
            
            'domain_adaptation': """
                Adapt the following generic prompt for the {domain} domain:
                
                Generic Prompt: {base_prompt}
                Domain Context: {domain_context}
                Domain Requirements: {domain_requirements}
                
                Requirements:
                1. Incorporate domain-specific terminology and concepts
                2. Add relevant context and background knowledge
                3. Adjust tone and complexity for domain experts
                4. Include domain-specific examples if helpful
                5. Ensure accuracy and appropriateness for {domain}
                
                Generate the domain-adapted prompt with reasoning.
            """,
            
            'few_shot_integration': """
                Enhance the following prompt with few-shot examples:
                
                Base Prompt: {base_prompt}
                Example Inputs/Outputs: {examples}
                Task Type: {task_type}
                
                Requirements:
                1. Integrate the most effective examples into the prompt
                2. Ensure examples demonstrate the desired behavior
                3. Format examples clearly and consistently
                4. Maintain prompt clarity and focus
                5. Use examples to guide AI behavior effectively
                
                Create an enhanced few-shot prompt with example integration strategy.
            """
        }
    
    async def generate_candidates(
        self,
        base_prompt: str,
        task_description: str,
        domain: Optional[str] = None,
        performance_goals: List[str] = None,
        baseline_examples: List[Dict[str, Any]] = None,
        constraints: Dict[str, Any] = None,
        num_candidates: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Generate multiple prompt candidates using meta-prompting.
        
        Args:
            base_prompt: Starting prompt template
            task_description: Description of the task
            domain: Target domain for specialization
            performance_goals: Optimization objectives
            baseline_examples: Example inputs/outputs
            constraints: Generation constraints
            num_candidates: Number of candidates to generate
            
        Returns:
            List of prompt candidates with metadata
        """
        try:
            self.logger.info(
                "Generating prompt candidates",
                task_description=task_description,
                domain=domain,
                num_candidates=num_candidates
            )
            
            start_time = time.time()
            candidates = []
            
            # Generate task-focused prompts
            task_candidates = await self._generate_task_focused_prompts(
                task_description=task_description,
                domain=domain or "general",
                performance_goals=performance_goals or ["accuracy"]
            )
            candidates.extend(task_candidates)
            
            # Generate domain-adapted prompts if domain specified
            if domain:
                domain_candidates = await self._generate_domain_adapted_prompts(
                    base_prompt=base_prompt,
                    domain=domain,
                    task_description=task_description
                )
                candidates.extend(domain_candidates)
            
            # Generate few-shot prompts if examples provided
            if baseline_examples:
                few_shot_candidates = await self._generate_few_shot_prompts(
                    base_prompt=base_prompt,
                    examples=baseline_examples,
                    task_description=task_description
                )
                candidates.extend(few_shot_candidates)
            
            # Generate constraint-aware prompts
            if constraints:
                constraint_candidates = await self._generate_constraint_aware_prompts(
                    base_prompt=base_prompt,
                    constraints=constraints,
                    task_description=task_description
                )
                candidates.extend(constraint_candidates)
            
            # Generate optimization-focused variants
            optimization_candidates = await self._generate_optimization_focused_prompts(
                base_prompt=base_prompt,
                performance_goals=performance_goals or ["accuracy"],
                task_description=task_description
            )
            candidates.extend(optimization_candidates)
            
            # Select best candidates
            selected_candidates = await self._select_best_candidates(
                candidates=candidates,
                num_candidates=num_candidates,
                performance_goals=performance_goals or ["accuracy"]
            )
            
            generation_time = time.time() - start_time
            
            # Add metadata to candidates
            for candidate in selected_candidates:
                candidate.update({
                    'generation_time': generation_time / len(selected_candidates),
                    'token_count': len(candidate['content'].split()),
                    'complexity_score': await self._calculate_complexity_score(candidate['content']),
                    'readability_score': await self._calculate_readability_score(candidate['content'])
                })
            
            self.logger.info(
                "Generated prompt candidates",
                num_candidates=len(selected_candidates),
                generation_time=generation_time
            )
            
            return selected_candidates
            
        except Exception as e:
            self.logger.error(
                "Failed to generate prompt candidates",
                error=str(e),
                task_description=task_description
            )
            raise
    
    async def refine_prompt(
        self,
        current_prompt: str,
        performance_feedback: Dict[str, Any],
        target_improvements: List[str],
        domain: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Refine an existing prompt based on performance feedback.
        
        Args:
            current_prompt: Current prompt to improve
            performance_feedback: Feedback on current performance
            target_improvements: Specific areas to improve
            domain: Target domain context
            
        Returns:
            Dict containing refined prompt and improvement reasoning
        """
        try:
            self.logger.info(
                "Refining prompt",
                current_prompt_length=len(current_prompt),
                target_improvements=target_improvements
            )
            
            # Analyze performance issues
            issues = await self._analyze_performance_issues(performance_feedback)
            
            # Generate refinement prompt
            refinement_prompt = self.meta_prompt_templates['refinement'].format(
                current_prompt=current_prompt,
                issues=issues,
                improvements=", ".join(target_improvements),
                domain=domain or "general"
            )
            
            # Generate refined prompt (simulated LLM call)
            refined_result = await self._simulate_llm_generation(
                prompt=refinement_prompt,
                max_length=self.max_prompt_length
            )
            
            # Extract and validate refined prompt
            refined_prompt = await self._extract_prompt_from_response(refined_result)
            
            # Calculate improvement metrics
            improvement_score = await self._calculate_improvement_potential(
                original=current_prompt,
                refined=refined_prompt,
                target_improvements=target_improvements
            )
            
            return {
                'content': refined_prompt,
                'method': 'iterative_refinement',
                'reasoning': f"Refined to address: {', '.join(target_improvements)}",
                'confidence': improvement_score,
                'improvements_addressed': target_improvements,
                'original_issues': issues,
                'parameters': {
                    'refinement_type': 'performance_feedback',
                    'domain': domain,
                    'feedback_metrics': list(performance_feedback.keys())
                }
            }
            
        except Exception as e:
            self.logger.error(
                "Failed to refine prompt",
                error=str(e),
                current_prompt_length=len(current_prompt)
            )
            raise
    
    async def adapt_to_domain(
        self,
        base_prompt: str,
        target_domain: str,
        domain_context: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Adapt a prompt for a specific domain.
        
        Args:
            base_prompt: Generic prompt to adapt
            target_domain: Target domain for adaptation
            domain_context: Additional domain context
            
        Returns:
            Dict containing domain-adapted prompt
        """
        try:
            self.logger.info(
                "Adapting prompt to domain",
                target_domain=target_domain,
                base_prompt_length=len(base_prompt)
            )
            
            # Get domain-specific requirements
            domain_requirements = await self._get_domain_requirements(target_domain)
            
            # Generate adaptation prompt
            adaptation_prompt = self.meta_prompt_templates['domain_adaptation'].format(
                base_prompt=base_prompt,
                domain=target_domain,
                domain_context=json.dumps(domain_context or {}),
                domain_requirements=domain_requirements
            )
            
            # Generate adapted prompt
            adapted_result = await self._simulate_llm_generation(
                prompt=adaptation_prompt,
                max_length=self.max_prompt_length
            )
            
            adapted_prompt = await self._extract_prompt_from_response(adapted_result)
            
            # Calculate domain fit score
            domain_fit_score = await self._calculate_domain_fit(
                prompt=adapted_prompt,
                domain=target_domain
            )
            
            return {
                'content': adapted_prompt,
                'method': 'domain_adaptation',
                'reasoning': f"Adapted for {target_domain} domain with specialized terminology and context",
                'confidence': domain_fit_score,
                'domain_adaptations': [target_domain],
                'parameters': {
                    'adaptation_type': 'domain_specialization',
                    'target_domain': target_domain,
                    'domain_requirements': domain_requirements
                }
            }
            
        except Exception as e:
            self.logger.error(
                "Failed to adapt prompt to domain",
                error=str(e),
                target_domain=target_domain
            )
            raise
    
    # Private helper methods
    
    async def _generate_task_focused_prompts(
        self,
        task_description: str,
        domain: str,
        performance_goals: List[str]
    ) -> List[Dict[str, Any]]:
        """Generate prompts focused on task performance."""
        candidates = []
        
        for i in range(3):  # Generate 3 task-focused variants
            # Simulate different approaches
            approaches = [
                "direct_instruction",
                "structured_reasoning", 
                "context_rich"
            ]
            
            approach = approaches[i % len(approaches)]
            
            # Generate prompt based on approach (simulated)
            if approach == "direct_instruction":
                content = f"Please {task_description.lower()}. Focus on {', '.join(performance_goals)}."
            elif approach == "structured_reasoning":
                content = f"""To {task_description.lower()}, follow these steps:
1. Analyze the input carefully
2. Consider the requirements: {', '.join(performance_goals)}
3. Provide a clear, accurate response
4. Verify your answer meets the criteria"""
            else:  # context_rich
                content = f"""You are an expert assistant specializing in {domain}. Your task is to {task_description.lower()}.

Please ensure your response optimizes for: {', '.join(performance_goals)}.

Consider the domain context and provide detailed, accurate information."""
            
            candidates.append({
                'content': content,
                'method': f'task_focused_{approach}',
                'reasoning': f'Generated using {approach} approach for task optimization',
                'confidence': 0.7 + (i * 0.1),
                'parameters': {
                    'approach': approach,
                    'task_focus': True,
                    'performance_goals': performance_goals
                }
            })
        
        return candidates
    
    async def _generate_domain_adapted_prompts(
        self,
        base_prompt: str,
        domain: str,
        task_description: str
    ) -> List[Dict[str, Any]]:
        """Generate domain-adapted prompt variants."""
        # Get domain-specific adaptations
        domain_terms = await self._get_domain_terminology(domain)
        domain_context = await self._get_domain_context(domain)
        
        adapted_prompt = f"""You are a {domain} specialist. {base_prompt}

Domain context: {domain_context}
Use appropriate {domain} terminology: {', '.join(domain_terms[:5])}

Ensure your response is accurate and appropriate for {domain} professionals."""
        
        return [{
            'content': adapted_prompt,
            'method': 'domain_adaptation',
            'reasoning': f'Adapted for {domain} domain with specialized terminology',
            'confidence': 0.8,
            'parameters': {
                'domain': domain,
                'domain_terms': domain_terms,
                'adaptation_type': 'terminology_and_context'
            }
        }]
    
    async def _generate_few_shot_prompts(
        self,
        base_prompt: str,
        examples: List[Dict[str, Any]],
        task_description: str
    ) -> List[Dict[str, Any]]:
        """Generate prompts with few-shot examples."""
        if not examples:
            return []
        
        # Format examples
        formatted_examples = []
        for i, example in enumerate(examples[:3]):  # Limit to 3 examples
            input_text = example.get('input', '')
            output_text = example.get('expected_output', '')
            formatted_examples.append(f"Example {i+1}:\nInput: {input_text}\nOutput: {output_text}")
        
        few_shot_prompt = f"""{base_prompt}

Here are some examples to guide your response:

{chr(10).join(formatted_examples)}

Now please respond to the following:"""
        
        return [{
            'content': few_shot_prompt,
            'method': 'few_shot_learning',
            'reasoning': f'Enhanced with {len(examples)} examples for better task understanding',
            'confidence': 0.85,
            'parameters': {
                'num_examples': len(examples),
                'example_integration': 'prefix_examples'
            }
        }]
    
    async def _generate_constraint_aware_prompts(
        self,
        base_prompt: str,
        constraints: Dict[str, Any],
        task_description: str
    ) -> List[Dict[str, Any]]:
        """Generate prompts that incorporate specific constraints."""
        constraint_text = []
        
        if 'max_length' in constraints:
            constraint_text.append(f"Keep your response under {constraints['max_length']} words")
        
        if 'style' in constraints:
            constraint_text.append(f"Use a {constraints['style']} tone and style")
        
        if 'format' in constraints:
            constraint_text.append(f"Format your response as: {constraints['format']}")
        
        if 'exclude_topics' in constraints:
            topics = constraints['exclude_topics']
            constraint_text.append(f"Avoid mentioning: {', '.join(topics)}")
        
        if constraint_text:
            constrained_prompt = f"""{base_prompt}

Please follow these constraints:
{chr(10).join(f"- {c}" for c in constraint_text)}"""
            
            return [{
                'content': constrained_prompt,
                'method': 'constraint_aware',
                'reasoning': f'Generated with {len(constraint_text)} specific constraints',
                'confidence': 0.75,
                'parameters': {
                    'constraints_applied': list(constraints.keys()),
                    'constraint_count': len(constraint_text)
                }
            }]
        
        return []
    
    async def _generate_optimization_focused_prompts(
        self,
        base_prompt: str,
        performance_goals: List[str],
        task_description: str
    ) -> List[Dict[str, Any]]:
        """Generate prompts optimized for specific performance goals."""
        candidates = []
        
        # Create goal-specific optimizations
        goal_optimizations = {
            'accuracy': "Focus on providing precise, factually correct information. Double-check your reasoning.",
            'token_efficiency': "Be concise and direct. Avoid unnecessary elaboration while maintaining clarity.",
            'user_satisfaction': "Provide helpful, clear responses that directly address the user's needs.",
            'clarity': "Use simple, clear language. Structure your response logically with good organization.",
            'relevance': "Stay focused on the specific question. Avoid tangential information.",
            'coherence': "Ensure your response flows logically from point to point with clear connections."
        }
        
        for goal in performance_goals:
            if goal in goal_optimizations:
                optimized_prompt = f"""{base_prompt}

{goal_optimizations[goal]}"""
                
                candidates.append({
                    'content': optimized_prompt,
                    'method': f'{goal}_optimization',
                    'reasoning': f'Optimized specifically for {goal}',
                    'confidence': 0.8,
                    'parameters': {
                        'optimization_goal': goal,
                        'optimization_type': 'goal_specific'
                    }
                })
        
        return candidates
    
    async def _select_best_candidates(
        self,
        candidates: List[Dict[str, Any]],
        num_candidates: int,
        performance_goals: List[str]
    ) -> List[Dict[str, Any]]:
        """Select the best candidates based on scoring criteria."""
        # Score candidates based on various factors
        for candidate in candidates:
            score = 0.0
            
            # Base confidence score
            score += candidate.get('confidence', 0.5) * 0.4
            
            # Method diversity bonus
            method_scores = {
                'task_focused': 0.3,
                'domain_adaptation': 0.25,
                'few_shot_learning': 0.35,
                'constraint_aware': 0.2,
                'optimization_focused': 0.3
            }
            
            method = candidate.get('method', '').split('_')[0]
            score += method_scores.get(method, 0.1) * 0.3
            
            # Performance goal alignment
            goal_alignment = 0.0
            for goal in performance_goals:
                if goal in candidate.get('method', '') or goal in candidate.get('reasoning', ''):
                    goal_alignment += 0.2
            score += min(goal_alignment, 0.3) * 0.3
            
            candidate['selection_score'] = score
        
        # Sort by score and return top candidates
        candidates.sort(key=lambda x: x.get('selection_score', 0), reverse=True)
        
        # Ensure diversity in selection
        selected = []
        used_methods = set()
        
        for candidate in candidates:
            method = candidate.get('method', '').split('_')[0]
            if len(selected) < num_candidates:
                if method not in used_methods or len(selected) >= num_candidates - 2:
                    selected.append(candidate)
                    used_methods.add(method)
        
        # Fill remaining slots if needed
        while len(selected) < num_candidates and len(selected) < len(candidates):
            for candidate in candidates:
                if candidate not in selected:
                    selected.append(candidate)
                    break
        
        return selected[:num_candidates]
    
    async def _simulate_llm_generation(
        self,
        prompt: str,
        max_length: int = 1000
    ) -> str:
        """Simulate LLM generation (placeholder for actual LLM integration)."""
        # In a real implementation, this would call an actual LLM API
        # For now, return a simulated response
        
        await asyncio.sleep(0.1)  # Simulate API call latency
        
        return f"""Based on the analysis, here is an optimized prompt:

[Generated optimized prompt that addresses the requirements and incorporates the specified improvements]

Reasoning: This prompt structure balances clarity with effectiveness, using domain-appropriate language and incorporating best practices for the specified performance goals."""
    
    async def _extract_prompt_from_response(self, response: str) -> str:
        """Extract the actual prompt from LLM response."""
        # Simple extraction logic (would be more sophisticated in practice)
        lines = response.split('\n')
        
        # Look for content between markers or after "prompt:"
        for i, line in enumerate(lines):
            if 'prompt:' in line.lower() or '[generated' in line.lower():
                # Return the next non-empty line or the line itself if it contains content
                if ':' in line and len(line.split(':', 1)[1].strip()) > 10:
                    return line.split(':', 1)[1].strip()
                elif i + 1 < len(lines):
                    return lines[i + 1].strip()
        
        # Fallback: return first substantial line
        for line in lines:
            if len(line.strip()) > 20:
                return line.strip()
        
        return response.strip()
    
    async def _calculate_complexity_score(self, prompt: str) -> float:
        """Calculate prompt complexity score."""
        # Simple complexity scoring based on various factors
        word_count = len(prompt.split())
        sentence_count = prompt.count('.') + prompt.count('!') + prompt.count('?')
        avg_word_length = sum(len(word) for word in prompt.split()) / max(word_count, 1)
        
        # Normalize to 0-1 scale
        complexity = min(1.0, (word_count / 200) * 0.4 + (avg_word_length / 8) * 0.3 + (sentence_count / 10) * 0.3)
        return complexity
    
    async def _calculate_readability_score(self, prompt: str) -> float:
        """Calculate prompt readability score."""
        # Simple readability scoring (inverse of complexity with adjustments)
        word_count = len(prompt.split())
        sentence_count = max(1, prompt.count('.') + prompt.count('!') + prompt.count('?'))
        avg_sentence_length = word_count / sentence_count
        
        # Prefer moderate sentence lengths
        readability = 1.0 - min(1.0, abs(avg_sentence_length - 15) / 20)
        return max(0.1, readability)
    
    async def _analyze_performance_issues(
        self, 
        performance_feedback: Dict[str, Any]
    ) -> List[str]:
        """Analyze performance feedback to identify specific issues."""
        issues = []
        
        for metric, score in performance_feedback.items():
            if isinstance(score, (int, float)) and score < 0.7:
                if metric == 'accuracy':
                    issues.append("Low accuracy - prompt may lack clarity or specificity")
                elif metric == 'relevance':
                    issues.append("Low relevance - prompt may be too broad or unfocused")
                elif metric == 'coherence':
                    issues.append("Low coherence - prompt structure may be unclear")
                elif metric == 'token_efficiency':
                    issues.append("Low token efficiency - prompt may be verbose or repetitive")
                else:
                    issues.append(f"Low {metric} - requires optimization")
        
        return issues or ["Performance optimization needed"]
    
    async def _calculate_improvement_potential(
        self,
        original: str,
        refined: str, 
        target_improvements: List[str]
    ) -> float:
        """Calculate potential improvement score for refined prompt."""
        # Simple scoring based on changes made
        length_ratio = len(refined) / max(len(original), 1)
        structure_changes = abs(refined.count('\n') - original.count('\n')) > 0
        content_changes = len(set(refined.split()) - set(original.split())) > 0
        
        improvement_score = 0.5  # Base score
        
        # Adjust based on target improvements
        for improvement in target_improvements:
            if improvement in ['clarity', 'coherence'] and structure_changes:
                improvement_score += 0.1
            elif improvement == 'token_efficiency' and length_ratio < 1.0:
                improvement_score += 0.15
            elif improvement == 'accuracy' and content_changes:
                improvement_score += 0.1
        
        return min(1.0, improvement_score)
    
    async def _get_domain_requirements(self, domain: str) -> str:
        """Get requirements for a specific domain."""
        domain_reqs = {
            'medical': "Use precise medical terminology, cite evidence-based practices, maintain professional tone",
            'legal': "Use accurate legal terminology, reference relevant laws/precedents, maintain formal tone",
            'technical': "Use precise technical language, include implementation details, focus on accuracy",
            'educational': "Use clear explanations, include examples, adapt to learning level",
            'business': "Use professional language, focus on practical outcomes, include relevant metrics",
            'creative': "Use engaging language, encourage creativity, allow for artistic expression"
        }
        
        return domain_reqs.get(domain.lower(), "Use appropriate terminology and maintain professional standards")
    
    async def _get_domain_terminology(self, domain: str) -> List[str]:
        """Get key terminology for a domain."""
        domain_terms = {
            'medical': ['diagnosis', 'treatment', 'symptoms', 'pathology', 'therapeutic', 'clinical'],
            'legal': ['statute', 'precedent', 'jurisdiction', 'liability', 'contract', 'tort'],
            'technical': ['implementation', 'architecture', 'optimization', 'debugging', 'framework', 'API'],
            'educational': ['pedagogy', 'curriculum', 'assessment', 'learning objectives', 'scaffolding'],
            'business': ['ROI', 'stakeholder', 'strategy', 'metrics', 'optimization', 'efficiency'],
            'creative': ['artistic', 'innovative', 'conceptual', 'aesthetic', 'narrative', 'expressive']
        }
        
        return domain_terms.get(domain.lower(), ['professional', 'specialized', 'expert', 'technical'])
    
    async def _get_domain_context(self, domain: str) -> str:
        """Get contextual background for a domain."""
        contexts = {
            'medical': "Healthcare and medical practice context with focus on patient care and evidence-based medicine",
            'legal': "Legal practice context with emphasis on accuracy, precedent, and regulatory compliance",
            'technical': "Software development and engineering context with focus on implementation and best practices",
            'educational': "Teaching and learning context with emphasis on student comprehension and engagement",
            'business': "Corporate and business context with focus on efficiency, profitability, and stakeholder value",
            'creative': "Artistic and creative context with emphasis on innovation, expression, and originality"
        }
        
        return contexts.get(domain.lower(), "Professional context with focus on accuracy and expertise")
    
    async def _calculate_domain_fit(self, prompt: str, domain: str) -> float:
        """Calculate how well a prompt fits a specific domain."""
        domain_terms = await self._get_domain_terminology(domain)
        
        # Count domain-specific terms in prompt
        prompt_words = set(prompt.lower().split())
        domain_term_count = sum(1 for term in domain_terms if term.lower() in prompt_words)
        
        # Calculate fit score
        domain_fit = min(1.0, domain_term_count / max(len(domain_terms), 1) + 0.3)
        return domain_fit