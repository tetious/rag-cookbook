"""
Agentic RAG - Interactive Generation
Interactive Q&A with the agentic RAG system.
"""

import os
from dotenv import load_dotenv

from agent import AgenticRAG

# Load environment variables
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


def interactive_mode():
    """Run interactive Q&A with the agent."""
    print("\n" + "=" * 60)
    print("🤖 Agentic RAG - Interactive Mode")
    print("=" * 60)
    print("\nThe agent will dynamically decide:")
    print("  • Whether to retrieve or use model knowledge")
    print("  • Which retrieval method to use")
    print("  • Whether to decompose complex questions")
    print("  • If results are sufficient or need retry")
    print("\nCommands:")
    print("  verbose:on/off  - Toggle detailed logging")
    print("  quit            - Exit")
    print("-" * 60)
    
    verbose = True
    agent = AgenticRAG(verbose=verbose)
    
    while True:
        print()
        user_input = input("❓ Your question: ").strip()
        
        if not user_input:
            continue
        
        if user_input.lower() in ['quit', 'exit', 'q']:
            print("\nGoodbye!")
            break
        
        if user_input.startswith('verbose:'):
            val = user_input.split(':')[1].lower()
            verbose = val in ['on', 'true', '1']
            agent = AgenticRAG(verbose=verbose)
            print(f"✅ Verbose mode: {'ON' if verbose else 'OFF'}")
            continue
        
        try:
            result = agent.run(user_input)
            
            print("\n" + "=" * 60)
            print("📝 ANSWER")
            print("=" * 60)
            print(result["answer"])
            
            print("\n📊 Agent Actions:")
            was_complex = result["analysis"].get("is_complex", False)
            needed_retrieval = result["analysis"].get("needs_retrieval", True)
            
            if not needed_retrieval:
                print("   → Decided retrieval not needed")
            elif was_complex:
                sub_qs = result["analysis"].get("sub_questions", [])
                print(f"   → Decomposed into {len(sub_qs)} sub-questions")
            
            for step in result["retrieval_steps"]:
                tool = step["tool"]
                docs = step["documents_retrieved"]
                print(f"   → {tool}: retrieved {docs} documents")
            
            print(f"   → Total: {result['total_documents']} documents used")
            
        except Exception as e:
            print(f"❌ Error: {e}")
            import traceback
            traceback.print_exc()


def main():
    """Run the interactive agent."""
    print("=" * 60)
    print("Agentic RAG - Dynamic Retrieval Agent")
    print("=" * 60)
    
    if not OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY not set")
    
    examples = [
        {
            "question": "What is insurance float?",
            "note": "Simple → likely single vector search"
        },
        {
            "question": "How did Apple contribute to Berkshire's portfolio in 2020?",
            "note": "Specific → filtered search (year + company)"
        },
        {
            "question": "Compare Berkshire's insurance performance in 2008 vs 2020",
            "note": "Complex → decomposed into sub-questions"
        },
        {
            "question": "What is 2+2?",
            "note": "Off-topic → may skip retrieval"
        },
    ]
    
    print("\n📋 Example Questions (showing agent behavior):")
    for i, ex in enumerate(examples, 1):
        print(f"  {i}. {ex['question']}")
        print(f"     💡 {ex['note']}")
    
    print("\n" + "-" * 60)
    choice = input("Enter 1-4 for examples, 'i' for interactive, or your question: ").strip()
    
    if choice.lower() == 'i':
        interactive_mode()
    elif choice in ['1', '2', '3', '4']:
        ex = examples[int(choice) - 1]
        print(f"\n📝 Question: {ex['question']}")
        print(f"💡 Expected: {ex['note']}")
        
        agent = AgenticRAG(verbose=True)
        result = agent.run(ex['question'])
        
        print("\n" + "=" * 60)
        print("📝 ANSWER")
        print("=" * 60)
        print(result["answer"])
    elif choice:
        agent = AgenticRAG(verbose=True)
        result = agent.run(choice)
        
        print("\n" + "=" * 60)
        print("📝 ANSWER")
        print("=" * 60)
        print(result["answer"])
    else:
        print("\n👋 No input. Run again!")


if __name__ == "__main__":
    main()
