"""
Evolutionary Optimizer using genetic algorithms for prompt optimization.

Implements genetic algorithms with mutation, crossover, selection, and
fitness evaluation to evolve high-performing prompt variants.
"""

import asyncio
import random
import time
import uuid
from typing import List, Dict, Any, Optional, Tuple, Set
from dataclasses import dataclass
from enum import Enum
import structlog
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, update

from ..models.prompt_optimization import PromptVariant, OptimizationExperiment, PromptEvaluation
from .performance_evaluator import PerformanceEvaluator

logger = structlog.get_logger()


class SelectionMethod(str, Enum):
    """Selection methods for evolutionary optimization."""
    TOURNAMENT = "tournament"
    ROULETTE_WHEEL = "roulette_wheel"
    RANK_BASED = "rank_based"
    ELITIST = "elitist"


class MutationStrategy(str, Enum):
    """Mutation strategies for prompt evolution."""
    RANDOM_WORD_REPLACEMENT = "random_word_replacement"
    SENTENCE_REORDERING = "sentence_reordering"
    INSTRUCTION_MODIFICATION = "instruction_modification"
    STYLE_ADAPTATION = "style_adaptation"
    STRUCTURAL_CHANGES = "structural_changes"


@dataclass
class Individual:
    """Represents an individual in the genetic algorithm population."""
    id: str
    content: str
    fitness_score: float
    generation: int
    parent_ids: List[str]
    mutation_history: List[str]
    evaluation_metrics: Dict[str, float]
    creation_method: str


class EvolutionaryOptimizer:
    """
    Genetic algorithm-based prompt optimization system.
    
    Features:
    - Multiple selection strategies (tournament, roulette wheel, rank-based)
    - Various mutation operators (word replacement, reordering, modifications)
    - Crossover operations for combining successful prompts
    - Elitism to preserve best individuals
    - Adaptive mutation rates based on population diversity
    - Multi-objective optimization support
    """
    
    def __init__(self, db_session: AsyncSession):
        self.db = db_session
        self.logger = logger.bind(component="evolutionary_optimizer")
        
        # Initialize performance evaluator
        self.performance_evaluator = PerformanceEvaluator(db_session)
        
        # Genetic algorithm parameters
        self.population_size = 20
        self.mutation_rate = 0.3
        self.crossover_rate = 0.7
        self.elitism_count = 2
        self.tournament_size = 3
        self.max_generations = 50
        self.convergence_threshold = 0.001
        self.diversity_threshold = 0.1
        
        # Mutation strategies and their weights
        self.mutation_strategies = {
            MutationStrategy.RANDOM_WORD_REPLACEMENT: 0.3,
            MutationStrategy.SENTENCE_REORDERING: 0.2,
            MutationStrategy.INSTRUCTION_MODIFICATION: 0.25,
            MutationStrategy.STYLE_ADAPTATION: 0.15,
            MutationStrategy.STRUCTURAL_CHANGES: 0.1
        }
        
        # Word pools for mutations
        self.enhancement_words = {
            'accuracy': ['precise', 'exact', 'specific', 'detailed', 'thorough'],
            'clarity': ['clear', 'simple', 'straightforward', 'understandable', 'explicit'],
            'efficiency': ['concise', 'brief', 'streamlined', 'focused', 'direct'],
            'creativity': ['innovative', 'creative', 'original', 'unique', 'imaginative'],
            'structure': ['organized', 'systematic', 'methodical', 'structured', 'logical']
        }
    
    async def optimize(
        self,
        experiment: OptimizationExperiment,
        max_iterations: Optional[int] = None,
        population_size: Optional[int] = None,
        test_cases: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """
        Run evolutionary optimization on a prompt.
        
        Args:
            experiment: The optimization experiment
            max_iterations: Maximum generations to run
            population_size: Size of the population
            test_cases: Test cases for evaluation
            
        Returns:
            Dict containing optimization results
        """
        try:
            start_time = time.time()
            
            # Override defaults if specified
            if max_iterations:
                self.max_generations = max_iterations
            if population_size:
                self.population_size = population_size
            
            self.logger.info(
                "Starting evolutionary optimization",
                experiment_id=str(experiment.id),
                population_size=self.population_size,
                max_generations=self.max_generations
            )
            
            # Initialize population
            initial_population = await self._initialize_population(experiment, test_cases)
            
            # Evolution loop
            population = initial_population
            generation_history = []
            best_individual = None
            baseline_score = 0.0
            
            for generation in range(self.max_generations):
                generation_start = time.time()
                
                # Evaluate population fitness
                population = await self._evaluate_population(population, test_cases)
                
                # Track best individual
                current_best = max(population, key=lambda x: x.fitness_score)
                if best_individual is None or current_best.fitness_score > best_individual.fitness_score:
                    best_individual = current_best
                
                # Record baseline score from first generation
                if generation == 0:
                    baseline_score = current_best.fitness_score
                
                # Calculate population statistics
                generation_stats = await self._calculate_population_stats(population, generation)
                generation_history.append(generation_stats)
                
                # Check convergence
                if await self._check_convergence(generation_history):
                    self.logger.info(
                        "Convergence achieved",
                        generation=generation,
                        best_score=best_individual.fitness_score
                    )
                    break
                
                # Update experiment progress
                progress = (generation + 1) / self.max_generations * 100
                await self._update_experiment_progress(experiment.id, progress, generation)
                
                # Create next generation
                population = await self._create_next_generation(population, experiment, generation + 1)
                
                generation_time = time.time() - generation_start
                self.logger.info(
                    "Generation completed",
                    generation=generation,
                    best_fitness=current_best.fitness_score,
                    avg_fitness=generation_stats['average_fitness'],
                    generation_time=generation_time
                )
            
            # Store best variant in database
            if best_individual:
                await self._store_best_variant(experiment, best_individual)
            
            optimization_time = time.time() - start_time
            
            result = {
                'best_score': best_individual.fitness_score if best_individual else 0.0,
                'baseline_score': baseline_score,
                'iterations_completed': len(generation_history),
                'convergence_achieved': len(generation_history) < self.max_generations,
                'optimization_time_seconds': optimization_time,
                'best_variant_id': best_individual.id if best_individual else None,
                'generation_history': generation_history,
                'final_population_diversity': generation_history[-1]['diversity'] if generation_history else 0.0,
                'improvement_percentage': (
                    ((best_individual.fitness_score - baseline_score) / baseline_score * 100)
                    if best_individual and baseline_score > 0 else 0.0
                ),
                'mutation_statistics': await self._get_mutation_statistics(generation_history),
                'selection_statistics': await self._get_selection_statistics(generation_history)
            }
            
            self.logger.info(
                "Evolutionary optimization completed",
                experiment_id=str(experiment.id),
                best_score=result['best_score'],
                improvement=result['improvement_percentage'],
                generations=result['iterations_completed']
            )
            
            return result
            
        except Exception as e:
            self.logger.error(
                "Evolutionary optimization failed",
                experiment_id=str(experiment.id),
                error=str(e)
            )
            raise
    
    async def _initialize_population(
        self,
        experiment: OptimizationExperiment,
        test_cases: Optional[List[Dict[str, Any]]]
    ) -> List[Individual]:
        """Initialize the starting population."""
        population = []
        base_prompt = experiment.base_prompt.template_content
        
        # Add the original prompt as first individual
        original = Individual(
            id=str(uuid.uuid4()),
            content=base_prompt,
            fitness_score=0.0,
            generation=0,
            parent_ids=[],
            mutation_history=[],
            evaluation_metrics={},
            creation_method="original"
        )
        population.append(original)
        
        # Generate variations
        for i in range(self.population_size - 1):
            variant_content = await self._generate_random_variant(base_prompt, i)
            
            variant = Individual(
                id=str(uuid.uuid4()),
                content=variant_content,
                fitness_score=0.0,
                generation=0,
                parent_ids=[original.id],
                mutation_history=[f"random_initialization_{i}"],
                evaluation_metrics={},
                creation_method="random_variation"
            )
            population.append(variant)
        
        return population
    
    async def _generate_random_variant(self, base_prompt: str, seed: int) -> str:
        """Generate a random variant of the base prompt."""
        # Set seed for reproducible randomness
        random.seed(seed)
        
        # Choose random mutation strategy
        strategies = list(self.mutation_strategies.keys())
        strategy = random.choice(strategies)
        
        # Apply mutation
        variant = await self._apply_mutation(base_prompt, strategy, intensity=0.3)
        
        return variant
    
    async def _evaluate_population(
        self,
        population: List[Individual],
        test_cases: Optional[List[Dict[str, Any]]]
    ) -> List[Individual]:
        """Evaluate fitness for all individuals in the population."""
        # Generate default test cases if none provided
        if not test_cases:
            test_cases = await self._generate_default_test_cases()
        
        # Evaluate each individual
        evaluation_tasks = []
        for individual in population:
            if individual.fitness_score == 0.0:  # Only evaluate if not already evaluated
                task = self._evaluate_individual_fitness(individual, test_cases)
                evaluation_tasks.append(task)
            else:
                evaluation_tasks.append(asyncio.create_task(self._return_individual(individual)))
        
        # Execute evaluations concurrently
        evaluated_population = await asyncio.gather(*evaluation_tasks)
        
        return evaluated_population
    
    async def _return_individual(self, individual: Individual) -> Individual:
        """Helper to return individual unchanged."""
        return individual
    
    async def _evaluate_individual_fitness(
        self,
        individual: Individual,
        test_cases: List[Dict[str, Any]]
    ) -> Individual:
        """Evaluate fitness score for an individual."""
        try:
            # Use performance evaluator to assess the prompt
            evaluation_result = await self.performance_evaluator.evaluate_prompt(
                prompt_content=individual.content,
                test_cases=test_cases,
                metrics=['accuracy', 'relevance', 'coherence', 'token_efficiency', 'clarity']
            )
            
            # Extract fitness score and detailed metrics
            individual.fitness_score = evaluation_result['performance_score']
            individual.evaluation_metrics = evaluation_result['detailed_metrics']
            
            return individual
            
        except Exception as e:
            self.logger.error(
                "Failed to evaluate individual fitness",
                individual_id=individual.id,
                error=str(e)
            )
            # Assign low fitness score on evaluation failure
            individual.fitness_score = 0.1
            individual.evaluation_metrics = {}
            return individual
    
    async def _create_next_generation(
        self,
        population: List[Individual],
        experiment: OptimizationExperiment,
        generation: int
    ) -> List[Individual]:
        """Create the next generation through selection, crossover, and mutation."""
        next_generation = []
        
        # Elitism: Keep best individuals
        sorted_population = sorted(population, key=lambda x: x.fitness_score, reverse=True)
        for i in range(self.elitism_count):
            elite = sorted_population[i]
            next_generation.append(Individual(
                id=str(uuid.uuid4()),
                content=elite.content,
                fitness_score=0.0,  # Reset for re-evaluation
                generation=generation,
                parent_ids=[elite.id],
                mutation_history=elite.mutation_history.copy(),
                evaluation_metrics={},
                creation_method="elitism"
            ))
        
        # Generate offspring
        while len(next_generation) < self.population_size:
            # Selection
            parent1 = await self._select_parent(population, SelectionMethod.TOURNAMENT)
            parent2 = await self._select_parent(population, SelectionMethod.TOURNAMENT)
            
            # Crossover
            if random.random() < self.crossover_rate:
                offspring_content = await self._crossover(parent1.content, parent2.content)
                creation_method = "crossover"
                parent_ids = [parent1.id, parent2.id]
                mutation_history = []
            else:
                offspring_content = parent1.content
                creation_method = "selection"
                parent_ids = [parent1.id]
                mutation_history = parent1.mutation_history.copy()
            
            # Mutation
            mutation_applied = False
            if random.random() < self.mutation_rate:
                mutation_strategy = await self._select_mutation_strategy()
                mutation_intensity = await self._calculate_adaptive_mutation_rate(population)
                offspring_content = await self._apply_mutation(
                    offspring_content, 
                    mutation_strategy,
                    mutation_intensity
                )
                mutation_history.append(f"{mutation_strategy.value}_gen_{generation}")
                creation_method += "_mutated"
                mutation_applied = True
            
            # Create offspring
            offspring = Individual(
                id=str(uuid.uuid4()),
                content=offspring_content,
                fitness_score=0.0,
                generation=generation,
                parent_ids=parent_ids,
                mutation_history=mutation_history,
                evaluation_metrics={},
                creation_method=creation_method
            )
            
            next_generation.append(offspring)
        
        return next_generation[:self.population_size]
    
    async def _select_parent(
        self,
        population: List[Individual],
        method: SelectionMethod
    ) -> Individual:
        """Select a parent individual using the specified method."""
        if method == SelectionMethod.TOURNAMENT:
            return await self._tournament_selection(population)
        elif method == SelectionMethod.ROULETTE_WHEEL:
            return await self._roulette_wheel_selection(population)
        elif method == SelectionMethod.RANK_BASED:
            return await self._rank_based_selection(population)
        else:  # Default to tournament
            return await self._tournament_selection(population)
    
    async def _tournament_selection(self, population: List[Individual]) -> Individual:
        """Tournament selection method."""
        tournament = random.sample(population, min(self.tournament_size, len(population)))
        return max(tournament, key=lambda x: x.fitness_score)
    
    async def _roulette_wheel_selection(self, population: List[Individual]) -> Individual:
        """Roulette wheel selection method."""
        total_fitness = sum(ind.fitness_score for ind in population)
        if total_fitness == 0:
            return random.choice(population)
        
        selection_point = random.uniform(0, total_fitness)
        current_sum = 0
        
        for individual in population:
            current_sum += individual.fitness_score
            if current_sum >= selection_point:
                return individual
        
        return population[-1]  # Fallback
    
    async def _rank_based_selection(self, population: List[Individual]) -> Individual:
        """Rank-based selection method."""
        sorted_population = sorted(population, key=lambda x: x.fitness_score)
        ranks = list(range(1, len(population) + 1))
        total_rank = sum(ranks)
        
        selection_point = random.uniform(0, total_rank)
        current_sum = 0
        
        for i, individual in enumerate(sorted_population):
            current_sum += ranks[i]
            if current_sum >= selection_point:
                return individual
        
        return sorted_population[-1]  # Fallback
    
    async def _crossover(self, parent1_content: str, parent2_content: str) -> str:
        """Perform crossover between two parent prompts."""
        # Simple sentence-level crossover
        sentences1 = [s.strip() for s in parent1_content.split('.') if s.strip()]
        sentences2 = [s.strip() for s in parent2_content.split('.') if s.strip()]
        
        if not sentences1 or not sentences2:
            return parent1_content
        
        # Take portions from both parents
        crossover_point = random.randint(1, min(len(sentences1), len(sentences2)) - 1)
        
        # Combine sentences
        offspring_sentences = sentences1[:crossover_point] + sentences2[crossover_point:]
        
        return '. '.join(offspring_sentences) + '.'
    
    async def _select_mutation_strategy(self) -> MutationStrategy:
        """Select a mutation strategy based on weights."""
        strategies = list(self.mutation_strategies.keys())
        weights = list(self.mutation_strategies.values())
        
        return random.choices(strategies, weights=weights)[0]
    
    async def _apply_mutation(
        self,
        content: str,
        strategy: MutationStrategy,
        intensity: float = 0.3
    ) -> str:
        """Apply a specific mutation strategy to the content."""
        try:
            if strategy == MutationStrategy.RANDOM_WORD_REPLACEMENT:
                return await self._mutate_word_replacement(content, intensity)
            elif strategy == MutationStrategy.SENTENCE_REORDERING:
                return await self._mutate_sentence_reordering(content, intensity)
            elif strategy == MutationStrategy.INSTRUCTION_MODIFICATION:
                return await self._mutate_instruction_modification(content, intensity)
            elif strategy == MutationStrategy.STYLE_ADAPTATION:
                return await self._mutate_style_adaptation(content, intensity)
            elif strategy == MutationStrategy.STRUCTURAL_CHANGES:
                return await self._mutate_structural_changes(content, intensity)
            else:
                return content
        except Exception as e:
            self.logger.error("Mutation failed", strategy=strategy.value, error=str(e))
            return content
    
    async def _mutate_word_replacement(self, content: str, intensity: float) -> str:
        """Replace random words with synonyms or enhancement words."""
        words = content.split()
        num_mutations = max(1, int(len(words) * intensity * 0.1))
        
        for _ in range(num_mutations):
            if not words:
                break
                
            # Choose random word to replace
            word_index = random.randint(0, len(words) - 1)
            original_word = words[word_index].lower().strip('.,!?')
            
            # Find appropriate replacement
            replacement = await self._find_word_replacement(original_word)
            if replacement:
                # Preserve capitalization and punctuation
                if words[word_index][0].isupper():
                    replacement = replacement.capitalize()
                
                # Preserve punctuation
                punct = ''.join(c for c in words[word_index] if not c.isalnum())
                words[word_index] = replacement + punct
        
        return ' '.join(words)
    
    async def _find_word_replacement(self, word: str) -> Optional[str]:
        """Find a suitable replacement word."""
        # Check enhancement word pools
        for category, word_list in self.enhancement_words.items():
            if word in ['good', 'nice', 'well'] and category == 'accuracy':
                return random.choice(word_list)
            elif word in ['easy', 'simple'] and category == 'clarity':
                return random.choice(word_list)
            elif word in ['short', 'quick'] and category == 'efficiency':
                return random.choice(word_list)
        
        # Basic synonyms for common words
        synonyms = {
            'make': ['create', 'generate', 'produce', 'develop'],
            'get': ['obtain', 'retrieve', 'acquire', 'gather'],
            'show': ['display', 'present', 'demonstrate', 'illustrate'],
            'use': ['utilize', 'employ', 'apply', 'implement'],
            'help': ['assist', 'support', 'aid', 'facilitate'],
            'find': ['locate', 'identify', 'discover', 'determine']
        }
        
        return random.choice(synonyms.get(word, [word]))[0] if word in synonyms else None
    
    async def _mutate_sentence_reordering(self, content: str, intensity: float) -> str:
        """Reorder sentences within the prompt."""
        sentences = [s.strip() for s in content.split('.') if s.strip()]
        
        if len(sentences) < 2:
            return content
        
        num_swaps = max(1, int(len(sentences) * intensity * 0.3))
        
        for _ in range(num_swaps):
            i, j = random.sample(range(len(sentences)), 2)
            sentences[i], sentences[j] = sentences[j], sentences[i]
        
        return '. '.join(sentences) + '.'
    
    async def _mutate_instruction_modification(self, content: str, intensity: float) -> str:
        """Modify instruction words and phrases."""
        # Replace instruction verbs
        instruction_replacements = {
            'please': ['kindly', 'ensure you', 'make sure to'],
            'analyze': ['examine', 'evaluate', 'assess', 'review'],
            'explain': ['describe', 'clarify', 'detail', 'outline'],
            'provide': ['give', 'supply', 'offer', 'present'],
            'create': ['generate', 'produce', 'develop', 'construct'],
            'consider': ['think about', 'take into account', 'evaluate', 'examine']
        }
        
        modified_content = content
        for original, replacements in instruction_replacements.items():
            if original in modified_content.lower() and random.random() < intensity:
                replacement = random.choice(replacements)
                modified_content = modified_content.replace(original, replacement, 1)
        
        return modified_content
    
    async def _mutate_style_adaptation(self, content: str, intensity: float) -> str:
        """Adapt the style and tone of the prompt."""
        # Add style modifiers
        style_additions = [
            "Please be thorough in your response.",
            "Focus on accuracy and precision.",
            "Use clear, professional language.",
            "Provide specific examples where appropriate.",
            "Ensure your response is well-structured."
        ]
        
        if random.random() < intensity:
            addition = random.choice(style_additions)
            return content + " " + addition
        
        return content
    
    async def _mutate_structural_changes(self, content: str, intensity: float) -> str:
        """Make structural changes to the prompt format."""
        if random.random() < intensity:
            # Add numbered structure
            if not any(char.isdigit() and char in content for char in '123'):
                sentences = [s.strip() for s in content.split('.') if s.strip()]
                if len(sentences) > 2:
                    # Convert to numbered list
                    numbered = []
                    for i, sentence in enumerate(sentences[:3], 1):
                        numbered.append(f"{i}. {sentence}")
                    return '. '.join(numbered) + '.'
        
        return content
    
    async def _calculate_adaptive_mutation_rate(self, population: List[Individual]) -> float:
        """Calculate adaptive mutation rate based on population diversity."""
        if len(population) < 2:
            return self.mutation_rate
        
        # Calculate diversity based on fitness variance
        fitness_scores = [ind.fitness_score for ind in population]
        fitness_variance = sum((x - sum(fitness_scores)/len(fitness_scores))**2 for x in fitness_scores) / len(fitness_scores)
        
        # Increase mutation rate if population is converging (low diversity)
        if fitness_variance < self.diversity_threshold:
            return min(0.8, self.mutation_rate * 1.5)
        else:
            return self.mutation_rate
    
    async def _calculate_population_stats(
        self,
        population: List[Individual],
        generation: int
    ) -> Dict[str, Any]:
        """Calculate statistics for the current population."""
        fitness_scores = [ind.fitness_score for ind in population]
        
        return {
            'generation': generation,
            'population_size': len(population),
            'best_fitness': max(fitness_scores),
            'worst_fitness': min(fitness_scores),
            'average_fitness': sum(fitness_scores) / len(fitness_scores),
            'fitness_std': (sum((x - sum(fitness_scores)/len(fitness_scores))**2 for x in fitness_scores) / len(fitness_scores)) ** 0.5,
            'diversity': await self._calculate_population_diversity(population),
            'creation_methods': {method: len([ind for ind in population if ind.creation_method == method]) 
                              for method in set(ind.creation_method for ind in population)}
        }
    
    async def _calculate_population_diversity(self, population: List[Individual]) -> float:
        """Calculate diversity measure for the population."""
        if len(population) < 2:
            return 0.0
        
        # Simple diversity based on unique content length
        unique_lengths = set(len(ind.content) for ind in population)
        length_diversity = len(unique_lengths) / len(population)
        
        # Fitness diversity
        fitness_scores = [ind.fitness_score for ind in population]
        fitness_range = max(fitness_scores) - min(fitness_scores)
        
        return (length_diversity + min(fitness_range, 1.0)) / 2.0
    
    async def _check_convergence(self, generation_history: List[Dict[str, Any]]) -> bool:
        """Check if the population has converged."""
        if len(generation_history) < 5:
            return False
        
        # Check if best fitness hasn't improved significantly in recent generations
        recent_best = [gen['best_fitness'] for gen in generation_history[-5:]]
        improvement = max(recent_best) - min(recent_best)
        
        return improvement < self.convergence_threshold
    
    async def _update_experiment_progress(
        self,
        experiment_id: uuid.UUID,
        progress: float,
        generation: int
    ) -> None:
        """Update experiment progress in database."""
        try:
            stmt = update(OptimizationExperiment).where(
                OptimizationExperiment.id == experiment_id
            ).values(
                progress_percentage=progress,
                current_iteration=generation
            )
            await self.db.execute(stmt)
            await self.db.commit()
        except Exception as e:
            self.logger.error("Failed to update experiment progress", error=str(e))
    
    async def _store_best_variant(
        self,
        experiment: OptimizationExperiment,
        best_individual: Individual
    ) -> None:
        """Store the best variant in the database."""
        try:
            variant = PromptVariant(
                experiment_id=experiment.id,
                parent_prompt_id=experiment.base_prompt_id,
                variant_content=best_individual.content,
                generation_method="evolutionary_optimization",
                generation_reasoning=f"Evolved over {best_individual.generation} generations using genetic algorithms",
                confidence_score=best_individual.fitness_score,
                iteration=best_individual.generation,
                parameters={
                    'creation_method': best_individual.creation_method,
                    'parent_ids': best_individual.parent_ids,
                    'mutation_history': best_individual.mutation_history,
                    'evaluation_metrics': best_individual.evaluation_metrics
                },
                ancestry=best_individual.parent_ids
            )
            
            self.db.add(variant)
            await self.db.commit()
            
        except Exception as e:
            self.logger.error("Failed to store best variant", error=str(e))
    
    async def _generate_default_test_cases(self) -> List[Dict[str, Any]]:
        """Generate default test cases for evaluation."""
        return [
            {
                'id': 'default_1',
                'input_data': {'text': 'Simple test query'},
                'expected_output': 'Clear, accurate response',
                'evaluation_criteria': {'accuracy': True, 'clarity': True}
            },
            {
                'id': 'default_2', 
                'input_data': {'text': 'Complex multi-part question'},
                'expected_output': 'Comprehensive structured response',
                'evaluation_criteria': {'completeness': True, 'coherence': True}
            },
            {
                'id': 'default_3',
                'input_data': {'text': 'Technical domain question'},
                'expected_output': 'Expert-level accurate response',
                'evaluation_criteria': {'accuracy': True, 'technical_depth': True}
            }
        ]
    
    async def _get_mutation_statistics(self, generation_history: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate mutation statistics across generations."""
        return {
            'total_mutations': len(generation_history) * self.population_size * self.mutation_rate,
            'mutation_rate': self.mutation_rate,
            'strategies_used': list(self.mutation_strategies.keys()),
            'adaptive_rates_used': True  # Simplified
        }
    
    async def _get_selection_statistics(self, generation_history: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate selection statistics across generations."""
        return {
            'selection_method': SelectionMethod.TOURNAMENT.value,
            'tournament_size': self.tournament_size,
            'elitism_count': self.elitism_count,
            'crossover_rate': self.crossover_rate
        }