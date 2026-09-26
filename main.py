from laya import Router


router = Router(preload=True)

state = {
    "body": "I was charged twice for the same order. Please refund one of the charges."
}

questions = {
    "department": {
        "type": "choice",
        "instructions": "Which department should handle this request?",
        "criteria": {
            "billing": "payments, refunds and invoices",
            "technical": "bugs and technical problems",
            "sales": "new purchases and pricing"
        }
    },

    "refund_requested": {
        "type": "noul",
        "instructions": "Does the user explicitly request a refund?"
    }
}

result = router.predict(state, questions)

print(result)