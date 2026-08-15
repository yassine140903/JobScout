"""~20 realistic fake job postings for offline testing. Mixed EN/FR, seniority, relevance."""

from __future__ import annotations

import hashlib

FIXTURE_POSTINGS: list[dict] = [
    # -- EN, Senior, Software Engineering --
    {
        "source": "fixture",
        "source_id": "fix-001",
        "url": "https://example.com/jobs/001",
        "title": "Senior Backend Engineer",
        "company": "Datastream AG",
        "description": (
            "We are looking for a senior backend engineer to design and maintain "
            "scalable microservices in Python and Go. You will work closely with "
            "the platform team on API design, database optimization, and CI/CD pipelines. "
            "5+ years experience required."
        ),
        "location": "Berlin, Germany",
        "country": "DE",
        "language": "en",
        "seniority": "senior",
        "posted_at": "2026-08-01",
    },
    # -- FR, Junior, Marketing --
    {
        "source": "fixture",
        "source_id": "fix-002",
        "url": "https://example.com/jobs/002",
        "title": "Chargé(e) de Marketing Digital",
        "company": "MaisonVerte",
        "description": (
            "Rejoignez notre équipe marketing pour gérer les campagnes digitales, "
            "le SEO/SEA et les réseaux sociaux. Poste junior ouvert aux jeunes diplômés. "
            "Bonne maîtrise de Google Analytics et des outils CRM souhaitée."
        ),
        "location": "Lyon, France",
        "country": "FR",
        "language": "fr",
        "seniority": "junior",
        "posted_at": "2026-07-28",
    },
    # -- EN, Mid, Data Science --
    {
        "source": "fixture",
        "source_id": "fix-003",
        "url": "https://example.com/jobs/003",
        "title": "Data Scientist",
        "company": "HealthQ Analytics",
        "description": (
            "Join our healthcare analytics team to build predictive models for patient "
            "outcomes. Strong Python, SQL, and statistics background required. "
            "Experience with scikit-learn or PyTorch is a plus. 2-4 years experience."
        ),
        "location": "Amsterdam, Netherlands",
        "country": "NL",
        "language": "en",
        "seniority": "mid",
        "posted_at": "2026-08-05",
    },
    # -- FR, Senior, Finance --
    {
        "source": "fixture",
        "source_id": "fix-004",
        "url": "https://example.com/jobs/004",
        "title": "Responsable Comptabilité",
        "company": "Groupe Financia",
        "description": (
            "Nous recherchons un(e) responsable comptabilité pour superviser l'équipe "
            "de 5 personnes, préparer les clôtures mensuelles et annuelles, et assurer "
            "la conformité fiscale. Expérience de 7 ans minimum en cabinet ou entreprise."
        ),
        "location": "Paris, France",
        "country": "FR",
        "language": "fr",
        "seniority": "senior",
        "posted_at": "2026-07-20",
    },
    # -- EN, Junior, Frontend --
    {
        "source": "fixture",
        "source_id": "fix-005",
        "url": "https://example.com/jobs/005",
        "title": "Junior Frontend Developer",
        "company": "PixelCraft Studios",
        "description": (
            "We're hiring a junior frontend developer to help build responsive web "
            "applications using React and TypeScript. You'll collaborate with designers "
            "and backend engineers. Recent graduates welcome."
        ),
        "location": "Dublin, Ireland",
        "country": "IE",
        "language": "en",
        "seniority": "junior",
        "posted_at": "2026-08-10",
    },
    # -- EN, Lead, DevOps --
    {
        "source": "fixture",
        "source_id": "fix-006",
        "url": "https://example.com/jobs/006",
        "title": "Lead DevOps Engineer",
        "company": "CloudNova",
        "description": (
            "Lead a team of 4 DevOps engineers managing Kubernetes clusters, "
            "Terraform infrastructure, and CI/CD pipelines across AWS and GCP. "
            "Strong experience with observability stacks (Prometheus, Grafana). "
            "8+ years in infrastructure or platform engineering."
        ),
        "location": "London, UK",
        "country": "GB",
        "language": "en",
        "seniority": "lead",
        "posted_at": "2026-08-03",
    },
    # -- FR, Mid, Mechanical Engineering --
    {
        "source": "fixture",
        "source_id": "fix-007",
        "url": "https://example.com/jobs/007",
        "title": "Ingénieur Mécanique",
        "company": "AeroPrecis",
        "description": (
            "Conception et simulation de pièces aéronautiques avec CATIA V5. "
            "Vous participerez aux revues de design et aux essais de qualification. "
            "Profil ingénieur avec 3-5 ans d'expérience dans le secteur aéronautique."
        ),
        "location": "Toulouse, France",
        "country": "FR",
        "language": "fr",
        "seniority": "mid",
        "posted_at": "2026-07-25",
    },
    # -- EN, Mid, Product Management --
    {
        "source": "fixture",
        "source_id": "fix-008",
        "url": "https://example.com/jobs/008",
        "title": "Product Manager — Payments",
        "company": "PayFlow",
        "description": (
            "Own the product roadmap for our payments platform. You'll work with "
            "engineering, design, and compliance to ship features used by 10M+ users. "
            "3+ years in product management, fintech experience preferred."
        ),
        "location": "Barcelona, Spain",
        "country": "ES",
        "language": "en",
        "seniority": "mid",
        "posted_at": "2026-08-07",
    },
    # -- EN, Senior, Research --
    {
        "source": "fixture",
        "source_id": "fix-009",
        "url": "https://example.com/jobs/009",
        "title": "Research Scientist — NLP",
        "company": "LangWorks Research",
        "description": (
            "Publish and productionize NLP research. Focus areas include multilingual "
            "transformers, retrieval-augmented generation, and evaluation methods. "
            "PhD in computational linguistics or related field. Strong publication record."
        ),
        "location": "Zurich, Switzerland",
        "country": "CH",
        "language": "en",
        "seniority": "senior",
        "posted_at": "2026-08-02",
    },
    # -- FR, Junior, HR --
    {
        "source": "fixture",
        "source_id": "fix-010",
        "url": "https://example.com/jobs/010",
        "title": "Assistant(e) Ressources Humaines",
        "company": "PeopleFirst",
        "description": (
            "En support de la DRH, vous gérerez l'administration du personnel, "
            "le suivi des absences et la préparation de la paie. "
            "Formation Bac+3 en RH, première expérience appréciée."
        ),
        "location": "Bordeaux, France",
        "country": "FR",
        "language": "fr",
        "seniority": "junior",
        "posted_at": "2026-08-09",
    },
    # -- EN, Senior, Cybersecurity --
    {
        "source": "fixture",
        "source_id": "fix-011",
        "url": "https://example.com/jobs/011",
        "title": "Senior Security Engineer",
        "company": "ShieldOps",
        "description": (
            "Design and implement security controls across cloud infrastructure. "
            "Incident response, threat modeling, and penetration testing. "
            "CISSP or equivalent certification preferred. 6+ years experience."
        ),
        "location": "Munich, Germany",
        "country": "DE",
        "language": "en",
        "seniority": "senior",
        "posted_at": "2026-07-30",
    },
    # -- EN, Junior, QA --
    {
        "source": "fixture",
        "source_id": "fix-012",
        "url": "https://example.com/jobs/012",
        "title": "QA Engineer",
        "company": "TestGrid",
        "description": (
            "Write and maintain automated test suites using Selenium and Pytest. "
            "You'll work with developers to improve test coverage and catch regressions. "
            "0-2 years experience. Strong attention to detail."
        ),
        "location": "Lisbon, Portugal",
        "country": "PT",
        "language": "en",
        "seniority": "junior",
        "posted_at": "2026-08-11",
    },
    # -- FR, Mid, Supply Chain --
    {
        "source": "fixture",
        "source_id": "fix-013",
        "url": "https://example.com/jobs/013",
        "title": "Analyste Supply Chain",
        "company": "LogiRoute",
        "description": (
            "Optimisation des flux logistiques et gestion des stocks. "
            "Vous analyserez les KPIs de la chaîne d'approvisionnement et proposerez "
            "des améliorations. Maîtrise d'Excel avancé et SAP. 3 ans d'expérience."
        ),
        "location": "Nantes, France",
        "country": "FR",
        "language": "fr",
        "seniority": "mid",
        "posted_at": "2026-07-22",
    },
    # -- EN, Principal, Architecture --
    {
        "source": "fixture",
        "source_id": "fix-014",
        "url": "https://example.com/jobs/014",
        "title": "Principal Software Architect",
        "company": "Nexion Systems",
        "description": (
            "Define technical strategy across a 200-person engineering org. "
            "Evaluate build-vs-buy, set standards for API design, data architecture, "
            "and system reliability. 12+ years with at least 4 in architecture roles."
        ),
        "location": "Stockholm, Sweden",
        "country": "SE",
        "language": "en",
        "seniority": "principal",
        "posted_at": "2026-08-04",
    },
    # -- EN, Mid, Mobile --
    {
        "source": "fixture",
        "source_id": "fix-015",
        "url": "https://example.com/jobs/015",
        "title": "iOS Developer",
        "company": "AppForge",
        "description": (
            "Build and maintain our flagship iOS app in Swift and SwiftUI. "
            "You'll own features end-to-end from design collaboration to App Store release. "
            "3+ years of iOS development. Experience with Core Data a plus."
        ),
        "location": "Copenhagen, Denmark",
        "country": "DK",
        "language": "en",
        "seniority": "mid",
        "posted_at": "2026-08-06",
    },
    # -- FR, Senior, Legal --
    {
        "source": "fixture",
        "source_id": "fix-016",
        "url": "https://example.com/jobs/016",
        "title": "Juriste Droit des Affaires",
        "company": "LexConseil",
        "description": (
            "Conseil juridique en droit commercial et droit des sociétés. "
            "Rédaction et négociation de contrats, suivi des contentieux. "
            "Master 2 en droit, 5 ans d'expérience minimum en cabinet ou entreprise."
        ),
        "location": "Paris, France",
        "country": "FR",
        "language": "fr",
        "seniority": "senior",
        "posted_at": "2026-07-18",
    },
    # -- EN, Junior, Data Engineering --
    {
        "source": "fixture",
        "source_id": "fix-017",
        "url": "https://example.com/jobs/017",
        "title": "Junior Data Engineer",
        "company": "PipelineIO",
        "description": (
            "Build ETL pipelines using Python, Airflow, and BigQuery. "
            "You'll help the data team move from batch to streaming architecture. "
            "Recent CS or data engineering graduates encouraged to apply."
        ),
        "location": "Vienna, Austria",
        "country": "AT",
        "language": "en",
        "seniority": "junior",
        "posted_at": "2026-08-08",
    },
    # -- EN, Mid, UX Design --
    {
        "source": "fixture",
        "source_id": "fix-018",
        "url": "https://example.com/jobs/018",
        "title": "UX Designer",
        "company": "FormFactor Design",
        "description": (
            "Lead user research and design for our B2B SaaS platform. "
            "Create wireframes, prototypes, and conduct usability testing. "
            "Proficiency in Figma required. 3-5 years experience in product design."
        ),
        "location": "Helsinki, Finland",
        "country": "FI",
        "language": "en",
        "seniority": "mid",
        "posted_at": "2026-08-12",
    },
    # -- FR, Mid, Teaching --
    {
        "source": "fixture",
        "source_id": "fix-019",
        "url": "https://example.com/jobs/019",
        "title": "Enseignant(e) en Informatique",
        "company": "Université de Strasbourg",
        "description": (
            "Enseignement en licence et master informatique. Cours d'algorithmique, "
            "bases de données et programmation Python. Doctorat requis. "
            "Expérience pédagogique souhaitée."
        ),
        "location": "Strasbourg, France",
        "country": "FR",
        "language": "fr",
        "seniority": "mid",
        "posted_at": "2026-07-15",
    },
    # -- EN, Senior, Biotech --
    {
        "source": "fixture",
        "source_id": "fix-020",
        "url": "https://example.com/jobs/020",
        "title": "Senior Bioinformatics Scientist",
        "company": "GenoLab",
        "description": (
            "Analyze genomic datasets and develop computational pipelines for drug "
            "discovery. Experience with R, Python, and NGS data analysis. "
            "PhD in bioinformatics or computational biology. 4+ years post-PhD."
        ),
        "location": "Basel, Switzerland",
        "country": "CH",
        "language": "en",
        "seniority": "senior",
        "posted_at": "2026-08-01",
    },
]


def get_fixtures() -> list[dict]:
    """Return fixture postings with url_hash and raw_data populated."""
    fixtures = []
    for posting in FIXTURE_POSTINGS:
        enriched = dict(posting)
        enriched["url_hash"] = hashlib.sha256(posting["url"].encode()).hexdigest()
        enriched["raw_data"] = None
        fixtures.append(enriched)
    return fixtures