import os
from dotenv import load_dotenv
load_dotenv()
import sys
import logging
from pathlib import Path


root_path = Path(__file__).resolve().parent.parent
sys.path.append(str(root_path))

from rag_lite.processing.chunking.chunker_controller import ChunkerController

os.environ["TOKENIZERS_PARALLELISM"] = "false"
logging.getLogger("transformers").setLevel(logging.ERROR)

def run_unified_test():
    print("\n" + "="*75)
    print(" RAG-LITE UNIFIED CHUNKING TEST (CONTEXT, TXT, MARKDOWN) ")
    print("="*75)

    # Initialize Controller
    try:
        tokenizer = os.getenv("MODEL_NAME")
        if tokenizer is None:
            raise ValueError("ERROR: La variable de entorno 'MODEL_NAME' no está definida. "
                    "Asegúrate de configurar tu archivo .env con la ruta al modelo.") 
        controller = ChunkerController(tokenizer_name=tokenizer)
        print("Chunker Controller initialized successfully.")
    except Exception as e:
        print(f"Initialization failed: {e}")
        return

    # Define Test Cases
    test_cases = [
        {
            "name": "CONVERSATIONAL CONTEXT (JSON List -> .context)",
            "ext": "context",
            "data": [
                {'role': 'user', 'content': 'Hola Pepe, ¿qué puedes hacer?'},
                {'role': 'assistant', 'content': 'Puedo ayudarte con:\n1. Responder preguntas.\n2. Analizar datos.\n3. Charlar un rato.'},
                {'role': 'user', 'content': 'Ando chill de cojones'}
            ]
        },
        {
            "name": "MARKDOWN DOCUMENT (Headers & Structure -> .md)",
            "ext": "md",
            "data": """
# POS_SYSTEM_TECHNICAL_CATALOG

---

### 1 TOOL: get_sales_summary
- **TOOL_NAME:** `get_sales_summary`
- **DESCRIPTION:** Generates a financial and operational report including total revenue, average ticket size, and payment methods (cash/card).
- **KEYWORDS:** revenue, income, total sales, financial summary, profit report, earnings.
- **ARGUMENTS:** `{"start_date": string, "end_date": string, "period": string}`
- **EXAMPLE_QUESTIONS:** 
  - "How much did we sell yesterday?"
  - "Give me the sales summary for this month."
  - "What was the total revenue for the last 3 days?"
- **JSON_FORMAT:** `{"tool": "get_sales_summary", "arguments": {"period": "ayer"}}`

---

### 2 TOOL: get_product_ranking
- **TOOL_NAME:** `get_product_ranking`
- **DESCRIPTION:** Returns a ranked list of products based on sales volume. Can show the best, the worst, or both.
- **KEYWORDS:** top products, best sellers, most sold, worst sellers, product ranking, least sold.
- **ARGUMENTS:** `{"limit": int, "criterion": "most"|"least"|"both", "period": string}`
- **EXAMPLE_QUESTIONS:** 
  - "What are the top 5 most sold products this week?"
  - "Show me the 10 products with the worst sales."
  - "Ranking of best sellers for last month."
- **JSON_FORMAT:** `{"tool": "get_product_ranking", "arguments": {"limit": 5, "criterion": "most", "period": "esta_semana"}}`

---

### 3 TOOL: get_low_stock
- **TOOL_NAME:** `get_low_stock`
- **DESCRIPTION:** Identifies products where inventory levels have dropped below a specific number.
- **KEYWORDS:** low stock, out of stock, restocking, inventory shortage, critical inventory.
- **ARGUMENTS:** `{"threshold": int}`
- **EXAMPLE_QUESTIONS:** 
  - "What products are running out?"
  - "Show me everything with stock less than 10 units."
  - "Is there anything I need to restock right now?"
- **JSON_FORMAT:** `{"tool": "get_low_stock", "arguments": {"threshold": 5}}`

---

### 4 TOOL: get_dead_inventory
- **TOOL_NAME:** `get_dead_inventory`
- **DESCRIPTION:** Finds products that have recorded zero sales since a specific reference date.
- **KEYWORDS:** dead stock, stagnant inventory, no sales, slow moving items, stuck products.
- **ARGUMENTS:** `{"reference_date": string}`
- **EXAMPLE_QUESTIONS:** 
  - "Which products haven't sold since last year?"
  - "Show me dead inventory since 2024-01-01."
  - "What items are stuck in the warehouse without movement?"
- **JSON_FORMAT:** `{"tool": "get_dead_inventory", "arguments": {"reference_date": "2024-01-01"}}`

---

### 5 TOOL: get_sales_velocity
- **TOOL_NAME:** `get_sales_velocity`
- **DESCRIPTION:** Calculates how fast a product sells per day and estimates how many days of stock are left.
- **KEYWORDS:** sales speed, depletion rate, burn rate, stockout estimate, days left.
- **ARGUMENTS:** `{"identifier": string, "period_days": int}`
- **EXAMPLE_QUESTIONS:** 
  - "When will we run out of Coca-Cola?"
  - "What is the sales velocity of SKU-100?"
  - "How many days of stock do I have left for this product?"
- **JSON_FORMAT:** `{"tool": "get_sales_velocity", "arguments": {"identifier": "SKU-99", "period_days": 30}}`

---

### 6 TOOL: get_inventory_valuation
- **TOOL_NAME:** `get_inventory_valuation`
- **DESCRIPTION:** Calculates the total monetary value of the inventory and projected profit margins.
- **KEYWORDS:** inventory value, warehouse worth, total assets, stock valuation, total cost.
- **ARGUMENTS:** `{"product_identifier": string}`
- **EXAMPLE_QUESTIONS:** 
  - "How much money is tied up in the inventory?"
  - "What is the total valuation of the warehouse?"
  - "What is the financial value of my current stock?"
- **JSON_FORMAT:** `{"tool": "get_inventory_valuation", "arguments": {}}`

---

###  7 TOOL: get_product_contribution
- **TOOL_NAME:** `get_product_contribution`
- **DESCRIPTION:** Measures the percentage of total company revenue generated by a specific item.
- **KEYWORDS:** sales share, revenue contribution, product impact, sales percentage.
- **ARGUMENTS:** `{"product_identifier": string, "period": string}`
- **EXAMPLE_QUESTIONS:** 
  - "How much does the iPhone contribute to total sales?"
  - "What is the revenue share of this product for last month?"
  - "Show me the sales impact of SKU-50."
- **JSON_FORMAT:** `{"tool": "get_product_contribution", "arguments": {"product_identifier": "A1", "period": "mes_pasado"}}`

---

### 8 TOOL: get_customer_sales
- **TOOL_NAME:** `get_customer_sales`
- **DESCRIPTION:** Analyzes a specific customer's behavior, including purchase history and favorite items.
- **KEYWORDS:** customer habits, purchase history, shopper behavior, favorite products, client spending.
- **ARGUMENTS:** `{"customer_id": int, "period": string}`
- **EXAMPLE_QUESTIONS:** 
  - "What has customer ID 450 bought recently?"
  - "Show me the favorite products of this shopper."
  - "What are the spending habits of client 123?"
- **JSON_FORMAT:** `{"tool": "get_customer_sales", "arguments": {"customer_id": 123, "period": "este_año"}}`

This is the **ANALYTICS_MODULE** documentation, normalized for RAG. Each chunk is designed to be instantly recognizable by the LLM, mapping natural language questions to technical JSON structures.

---


### 9 TOOL: get_order_detail
- **TOOL_NAME:** `get_order_detail`
- **DESCRIPTION:** Retrieves the complete itemized breakdown of a specific transaction, including prices, quantities, and totals.
- **KEYWORDS:** ticket details, order info, specific sale, item breakdown, transaction content.
- **ARGUMENTS:** `{"order_id": int}`
- **EXAMPLE_QUESTIONS:** 
  - "Show me the details for order ID 501."
  - "What was sold in ticket #45?"
  - "Give me the breakdown for order 1024."
- **JSON_FORMAT:** `{"tool": "get_order_detail", "arguments": {"order_id": 501}}`

---

### 10 TOOL: search_recent_orders
- **TOOL_NAME:** `search_recent_orders`
- **DESCRIPTION:** Searches for transactions using filters like folio number or payment status. Can return a general list of the latest sales.
- **KEYWORDS:** recent sales, folio search, pending orders, paid tickets, cancelled sales, transaction history.
- **CONSTRAINTS:** `status` must be: "PENDING", "PAID", or "CANCELLED".
- **ARGUMENTS:** `{"ticket_folio": string, "status": string, "limit": int}`
- **EXAMPLE_QUESTIONS:** 
  - "Show me the last 10 sales."
  - "Search for folio number A-102."
  - "List all pending orders."
  - "Find the most recent cancelled tickets."
- **JSON_FORMAT:** `{"tool": "search_recent_orders", "arguments": {"status": "PAID", "limit": 5}}`

This is the **PRODUCTS_MODULE** documentation, normalized for RAG. It covers the catalog, specific product lookups, technical specifications, and active discounts.

---

### 11 TOOL: get_all_products
- **TOOL_NAME:** `get_all_products`
- **DESCRIPTION:** Retrieves the entire company product catalog, including names, current prices, total stock, and SKUs.
- **KEYWORDS:** product catalog, price list, all items, inventory list, stock check.
- **ARGUMENTS:** `{}`
- **EXAMPLE_QUESTIONS:** 
  - "Show me the full price list."
  - "What products do we have in the catalog?"
  - "List all items and their current stock."
- **JSON_FORMAT:** `{"tool": "get_all_products", "arguments": {}}`

---

### 12 TOOL: get_all_promotions
- **TOOL_NAME:** `get_all_promotions`
- **DESCRIPTION:** Retrieves a list of all currently active discounts, sales, and special offers across the store.
- **KEYWORDS:** active offers, store discounts, sales, current promos, price drops.
- **ARGUMENTS:** `{}`
- **EXAMPLE_QUESTIONS:** 
  - "What are the current promotions?"
  - "Are there any active discounts today?"
  - "Show me all products on sale."
- **JSON_FORMAT:** `{"tool": "get_all_promotions", "arguments": {}}`

---

### 13 TOOL: get_promotions_by_product
- **TOOL_NAME:** `get_promotions_by_product`
- **DESCRIPTION:** Checks if a specific product (by ID) has an active discount or is part of a special offer.
- **KEYWORDS:** check discount, item sale, product promo, discount status.
- **ARGUMENTS:** `{"product_id": int}`
- **EXAMPLE_QUESTIONS:** 
  - "Does product ID 45 have a discount?"
  - "Check if there's an offer for this item."
  - "Is this product on sale right now?"
- **JSON_FORMAT:** `{"tool": "get_promotions_by_product", "arguments": {"product_id": 45}}`

---

### 14 TOOL: get_product_by_id
- **TOOL_NAME:** `get_product_by_id`
- **DESCRIPTION:** Returns detailed technical information, supplier data, SKU, and quantities currently reserved for pending orders.
- **KEYWORDS:** technical specs, product details, supplier info, reserved stock, item ID lookup.
- **ARGUMENTS:** `{"product_id": int}`
- **EXAMPLE_QUESTIONS:** 
  - "Technical details for product 101."
  - "Who is the supplier for this item?"
  - "How much stock is reserved for this product?"
- **JSON_FORMAT:** `{"tool": "get_product_by_id", "arguments": {"product_id": 101}}`

---

### 15 TOOL: search_product_by_sku
- **TOOL_NAME:** `search_product_by_sku`
- **DESCRIPTION:** Direct search for a product using its SKU (Stock Keeping Unit) string. Ideal for quick code lookups.
- **KEYWORDS:** SKU search, barcode lookup, product code, find by SKU.
- **ARGUMENTS:** `{"sku": string}`
- **EXAMPLE_QUESTIONS:** 
  - "Search for SKU A-450-B."
  - "Find the product with code 12345."
  - "Which product belongs to SKU 'COKE-01'?"
- **JSON_FORMAT:** `{"tool": "search_product_by_sku", "arguments": {"sku": "A-450-B"}}`

This is the **CUSTOMERS_MODULE** documentation, normalized for RAG. It covers the shopper directory, loyalty programs, credit management, and personal profiles.

---

### 16 TOOL: get_all_customers
- **TOOL_NAME:** `get_all_customers`
- **DESCRIPTION:** Retrieves the full list of all registered customers, including frequent shoppers and members.
- **KEYWORDS:** customer list, shopper directory, member base, frequent shoppers, client base.
- **ARGUMENTS:** `{}`
- **EXAMPLE_QUESTIONS:** 
  - "Show me all registered customers."
  - "List our frequent shoppers."
  - "Who are the clients in our database?"
- **JSON_FORMAT:** `{"tool": "get_all_customers", "arguments": {}}`

---

### 17 TOOL: get_customer_points_history
- **TOOL_NAME:** `get_customer_points_history`
- **DESCRIPTION:** Retrieves the loyalty points balance, redemption history, and point accumulation records for a specific customer.
- **KEYWORDS:** loyalty points, reward balance, point redemption, points history, customer rewards.
- **ARGUMENTS:** `{"customer_id": int}`
- **EXAMPLE_QUESTIONS:** 
  - "How many points does customer 101 have?"
  - "Show me the points redemption history for client 45."
  - "What is the rewards balance for shopper ID 99?"
- **JSON_FORMAT:** `{"tool": "get_customer_points_history", "arguments": {"customer_id": 101}}`

---

### 18 TOOL: get_customer_credit_history
- **TOOL_NAME:** `get_customer_credit_history`
- **DESCRIPTION:** Provides a detailed financial history of a customer, including current debt (amount owed), available credit limits, and past payment records.
- **KEYWORDS:** customer debt, credit limit, amount owed, credit balance, payment records, available credit.
- **ARGUMENTS:** `{"customer_id": int}`
- **EXAMPLE_QUESTIONS:** 
  - "How much does customer 88 owe?"
  - "Show the credit history for client 10."
  - "What is the available credit limit for ID 45?"
- **JSON_FORMAT:** `{"tool": "get_customer_credit_history", "arguments": {"customer_id": 88}}`

---

### 19 TOOL: get_customer_detail
- **TOOL_NAME:** `get_customer_detail`
- **DESCRIPTION:** Retrieves the full personal profile of a customer, including contact email, demographic information, and date of birth.
- **KEYWORDS:** customer profile, personal info, contact email, birthday, demographic data, client details.
- **ARGUMENTS:** `{"customer_id": int}`
- **EXAMPLE_QUESTIONS:** 
  - "Get the profile for customer 123."
  - "What is the contact email for client 45?"
  - "Show me the demographic data for user ID 10."
- **JSON_FORMAT:** `{"tool": "get_customer_detail", "arguments": {"customer_id": 123}}`

This is the **SUPPLIERS_MODULE** documentation, normalized for RAG. It covers the vendor directory and specific administrative/tax details of business partners.

---

### 20 TOOL: get_all_suppliers
- **TOOL_NAME:** `get_all_suppliers`
- **DESCRIPTION:** Retrieves the complete directory of all registered suppliers and their general information.
- **KEYWORDS:** supplier list, vendor directory, wholesale partners, list all suppliers.
- **ARGUMENTS:** `{}`
- **EXAMPLE_QUESTIONS:** 
  - "Who are all our suppliers?"
  - "Show me the list of registered vendors."
  - "Give me the directory of wholesale partners."
- **JSON_FORMAT:** `{"tool": "get_all_suppliers", "arguments": {}}`

---

### 21 TOOL: get_supplier_detail
- **TOOL_NAME:** `get_supplier_detail`
- **DESCRIPTION:** Retrieves the full profile of a specific supplier, including tax identification (RFC/Tax ID), contact email, phone number, and physical address.
- **KEYWORDS:** supplier contact, vendor Tax ID, RFC, supplier address, vendor phone, supplier profile.
- **ARGUMENTS:** `{"supplier_id": int}`
- **EXAMPLE_QUESTIONS:** 
  - "What is the Tax ID (RFC) for supplier 10?"
  - "Show me the contact details and address for vendor 4."
  - "Get the profile for supplier ID 12."
- **JSON_FORMAT:** `{"tool": "get_supplier_detail", "arguments": {"supplier_id": 10}}`

This is the **CHATBOT_USERS_MODULE** documentation, normalized for RAG. It focuses on system security, access whitelists, and user verification.

---

### 22 TOOL: get_all_chatbot_users
- **TOOL_NAME:** `get_all_chatbot_users`
- **DESCRIPTION:** Retrieves the complete whitelist of authorized personnel who have permission to interact with the chatbot system.
- **KEYWORDS:** whitelist, bot users, authorized personnel, access list, chatbot permissions.
- **ARGUMENTS:** `{}`
- **EXAMPLE_QUESTIONS:** 
  - "Who is allowed to use this bot?"
  - "Show me the whitelist of authorized users."
  - "List all personnel with chatbot access."
- **JSON_FORMAT:** `{"tool": "get_all_chatbot_users", "arguments": {}}`

---

### 23 TOOL: get_chatbot_user
- **TOOL_NAME:** `get_chatbot_user`
- **DESCRIPTION:** Verifies the specific authorization status and retrieves the last connection details for a user identified by their mobile phone number.
- **KEYWORDS:** verify access, user status check, mobile number lookup, connection details, permission verify.
- **ARGUMENTS:** `{"mobile_number": string}`
- **EXAMPLE_QUESTIONS:** 
  - "Does the number +1234567890 have access?"
  - "Check the last connection for mobile 555-0199."
  - "Verify if this phone number is in the authorized list."
- **JSON_FORMAT:** `{"tool": "get_chatbot_user", "arguments": {"mobile_number": "+1234567890"}}`

This is the **SYSTEM_TOOLS_MODULE** documentation, normalized for RAG. These tools are the most critical, as they allow the agent to manage its own memory and retrieve the other documentation chunks you've just created.

---

### 24 TOOL: fetch_chat_history
- **TOOL_NAME:** `fetch_chat_history`
- **DESCRIPTION:** Retrieves the most recent messages from the current active conversation. Use this to resolve pronouns (e.g., "it", "him", "that") or to recall immediate previous instructions.
- **KEYWORDS:** chat history, previous messages, conversation log, context, memory, what was said, repeat that.
- **ARGUMENTS:** `{"telegram_id": int, "limit": int}`
- **EXAMPLE_QUESTIONS:** 
  - "What did I just say?"
  - "Repeat the last thing you told me."
  - "What was the price of the product we were talking about?"
  - "Check the previous messages for the customer ID."
- **JSON_FORMAT:** `{"tool": "fetch_chat_history", "arguments": {"telegram_id": 6596706525, "limit": 5}}`

---

### 25 TOOL: search_system_context
- **TOOL_NAME:** `search_system_context`
- **DESCRIPTION:** Searches the long-term knowledge base (manuals, API documentation, tool schemas) and past user interactions. Use this when you don't know how a tool works or need to find a specific store policy.
- **KEYWORDS:** search manual, documentation, how to use, business rules, system info, background context, find policy.
- **ARGUMENTS:** `{"query": string, "telegram_id": int}`
- **EXAMPLE_QUESTIONS:** 
  - "How do I check the sales summary?"
  - "What is the tool to see customer debt?"
  - "Find the documentation for inventory valuation."
  - "What did the user want to buy in their last session months ago?"
- **JSON_FORMAT:** `{"tool": "search_system_context", "arguments": {"query": "get_sales_summary documentation", "telegram_id": 6596706525}}`

---

            """ #* 3 # Repeat to force multiple chunks
        },
        {
            "name": "PLAIN TEXT (Standard Prose -> .txt)",
            "ext": "txt",
            "data": """
            The e5-small-v2 model is a highly efficient text encoder. It is specifically designed for information retrieval tasks and semantic search. 
            Retrieval-Augmented Generation (RAG) systems benefit significantly from having coherent text fragments. 
            If we cut a sentence in the middle, we lose semantic meaning.
            """ * 6
        }
    ]

    # Export Path
    export_file = root_path / "debug_chunks_unified.txt"
    
    with open(export_file, "w", encoding="utf-8") as f:
        f.write("=== RAG-LITE UNIFIED DEBUG EXPORT ===\n\n")

        for case in test_cases:
            print(f"\n>>> Processing: {case['name']}")
            
            # The controller automatically chooses the strategy based on the extension
            chunks = controller.process(case['data'], extension=case['ext'])
            
            f.write(f"--- TEST CASE: {case['name']} ---\n")
            f.write(f"Total Chunks: {len(chunks)}\n\n")

            for i, chunk in enumerate(chunks):
                tokens = controller.chunker._length_function(chunk)
                
                # Console Output (Minimal Preview)
                header = f" CHUNK #{i+1} ({tokens} tokens) "
                print(f"\033[93m{header:-^60}\033[0m")
                # Show first 100 chars and a bit of the end to see the cut
                preview = f"{chunk.strip()[:100]} [...] {chunk.strip()[-40:]}"
                print(preview.replace('\n', ' ')) 
                
                # File Output (Full Content for auditing)
                f.write(f"CHUNK #{i+1} [{tokens} tokens]:\n{chunk.strip()}\n")
                f.write("-" * 40 + "\n\n")
            
            f.write("\n" + "="*60 + "\n\n")

    print("\n" + "="*75)
    print(f"UNIFIED TEST COMPLETE")
    print(f"Detailed results saved to: {export_file}")
    print("="*75)

if __name__ == "__main__":
    run_unified_test()