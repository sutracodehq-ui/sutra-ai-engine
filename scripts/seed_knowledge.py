"""
Seed Knowledge — Populate Qdrant with initial brand knowledge.

Run this once to give the AI some foundational knowledge to work with.
After running, the chatbot will answer questions using this data.

Usage:
    python scripts/seed_knowledge.py --brand-id 1
"""

import argparse
import asyncio
import sys

sys.path.insert(0, ".")

from app.services.intelligence.memory import get_memory


# ─── Default Knowledge Base (Vidyantra / EdTech) ────────────

VIDYANTRA_FAQ = [
    {
        "question": "What is Vidyantra?",
        "answer": "Vidyantra is an AI-powered EdTech platform that helps schools, coaching centers, and educational institutions manage their operations efficiently. It includes modules for attendance tracking, fee management, timetable scheduling, exam results, parent communication, and AI-powered insights."
    },
    {
        "question": "What features does Vidyantra offer?",
        "answer": "Vidyantra offers: (1) Attendance Management — digital attendance with reports, (2) Fee Management — track payments, send reminders, generate receipts, (3) Timetable — auto-generate and manage class schedules, (4) Exam & Results — enter marks, calculate grades, generate report cards, (5) Parent App — real-time updates for parents, (6) AI Assistant — smart chatbot for queries, quiz generation, and educational content, (7) Staff Management — track teachers, leaves, and performance, (8) UDISE+ Compliance — auto-fill government reporting forms."
    },
    {
        "question": "How much does Vidyantra cost?",
        "answer": "Vidyantra offers flexible pricing based on institution size. Contact our team for a personalized quote. We offer free demos and pilot programs for schools that want to try before committing."
    },
    {
        "question": "Does Vidyantra support Hindi and regional languages?",
        "answer": "Yes! Vidyantra fully supports Hindi, English, and Hinglish. Our AI assistant can understand and respond in Indian languages. We're expanding to more regional languages soon."
    },
    {
        "question": "How do I get started with Vidyantra?",
        "answer": "Getting started is easy: (1) Request a demo at our website, (2) Our team will set up your institution's account, (3) Import your student and staff data, (4) Start using the platform. We provide full onboarding support and training."
    },
    {
        "question": "Can Vidyantra generate quizzes and educational content?",
        "answer": "Yes! Vidyantra has AI-powered content generation. You can generate MCQ quizzes, study notes, flashcards, and lesson plans. Just tell the AI the topic, class level, and difficulty, and it creates structured educational content instantly."
    },
    {
        "question": "Is Vidyantra compliant with NEP 2020?",
        "answer": "Yes, Vidyantra is designed with NEP 2020 guidelines in mind. It supports competency-based assessment, experiential learning tracking, holistic report cards (360° progress), and multilingual education — all key requirements of the National Education Policy."
    },
    {
        "question": "Does Vidyantra have a mobile app for parents?",
        "answer": "Yes! Parents get a dedicated mobile app where they can view attendance, exam results, fee status, school announcements, and communicate with teachers. Push notifications keep them updated in real-time."
    },
    {
        "question": "Can Vidyantra handle multiple branches?",
        "answer": "Absolutely. Vidyantra supports multi-branch management from a single dashboard. You can manage multiple schools or branches with centralized reporting and individual branch-level controls."
    },
    {
        "question": "What kind of support does Vidyantra provide?",
        "answer": "We provide: (1) 24/7 AI chatbot support, (2) Dedicated account manager, (3) Phone and WhatsApp support during business hours, (4) Video training and onboarding, (5) Regular updates and new feature releases."
    },
]

# ─── General Education Knowledge ────────────────────────────

EDUCATION_KB = [
    {
        "question": "What is NEP 2020?",
        "answer": "The National Education Policy 2020 is India's comprehensive education reform framework. Key changes: 5+3+3+4 school structure (replacing 10+2), emphasis on mother tongue instruction till Class 5, coding from Class 6, vocational training from Class 6, reformed board exams focusing on core competencies, 360-degree holistic report cards, and a target of 50% GER in higher education by 2035."
    },
    {
        "question": "What is the RTE Act?",
        "answer": "The Right to Education Act (RTE) 2009 makes free and compulsory education a fundamental right for children aged 6-14. Key provisions: 25% reservation for economically weaker sections in private schools, minimum infrastructure standards, pupil-teacher ratios (1:30 for primary, 1:35 for upper primary), no detention policy till Class 8, and prohibition of physical punishment."
    },
    {
        "question": "What is UDISE+?",
        "answer": "UDISE+ (Unified District Information System for Education Plus) is the Indian government's school data collection platform. All schools must submit annual data covering: student enrollment, infrastructure, teachers, facilities, and academic outcomes. It's mandatory for CBSE, ICSE, and State Board affiliated schools."
    },
]


async def seed(brand_id: str):
    """Seed the knowledge base for a brand (Qdrant via Memory)."""
    mem = get_memory()

    print(f"Seeding knowledge for brand {brand_id}...")
    result = await mem.brand_import_faq(brand_id, VIDYANTRA_FAQ)
    print(f"  ✅ FAQ: {result['imported']}/{result['total']} items imported")

    result = await mem.brand_import_faq(brand_id, EDUCATION_KB)
    print(f"  ✅ Education KB: {result['imported']}/{result['total']} items imported")

    from app.services.vector.qdrant_store import get_qdrant_client, qdrant_collection_count

    client = get_qdrant_client()
    total = qdrant_collection_count(client, f"brand_{brand_id}_knowledge") if client else 0
    print(f"\n  📊 Total knowledge points in Qdrant: {total}")
    print("  Done! The AI will now use this knowledge to answer questions.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Seed Qdrant with initial brand knowledge")
    parser.add_argument("--brand-id", default="1", help="Brand/tenant ID to seed (default: 1)")
    args = parser.parse_args()

    asyncio.run(seed(args.brand_id))
