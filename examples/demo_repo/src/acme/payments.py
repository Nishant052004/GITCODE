"""Invoices, charging and refunds against a (stubbed) payment gateway."""
import uuid
from dataclasses import dataclass, field


class PaymentError(Exception):
    """Raised when the gateway declines a charge or a refund."""


@dataclass
class Invoice:
    invoice_id: str
    customer_email: str
    amount_cents: int
    currency: str = "usd"
    paid: bool = False
    line_items: list = field(default_factory=list)


def create_invoice(customer_email: str, amount_cents: int, currency: str = "usd") -> Invoice:
    """Open a new unpaid invoice for a customer."""
    if amount_cents <= 0:
        raise ValueError("amount_cents must be positive")
    return Invoice(
        invoice_id=f"inv_{uuid.uuid4().hex[:10]}",
        customer_email=customer_email,
        amount_cents=amount_cents,
        currency=currency,
    )


def charge_invoice(invoice: Invoice, gateway, idempotency_key: str | None = None) -> Invoice:
    """Charge an invoice through the gateway.

    Retries are handled by the caller; this function is idempotent for a given
    idempotency_key so duplicated webhooks never double-charge a customer.
    Raises PaymentError when the gateway declines.
    """
    if invoice.paid:
        return invoice
    key = idempotency_key or f"charge-{invoice.invoice_id}"
    result = gateway.charge(
        amount_cents=invoice.amount_cents,
        currency=invoice.currency,
        customer=invoice.customer_email,
        idempotency_key=key,
    )
    if not result.get("ok"):
        raise PaymentError(result.get("error", "gateway declined the charge"))
    invoice.paid = True
    return invoice


def refund_invoice(invoice: Invoice, gateway, amount_cents: int | None = None) -> dict:
    """Refund a paid invoice (fully by default)."""
    if not invoice.paid:
        raise PaymentError("cannot refund an unpaid invoice")
    amount = amount_cents or invoice.amount_cents
    result = gateway.refund(invoice_id=invoice.invoice_id, amount_cents=amount)
    if not result.get("ok"):
        raise PaymentError(result.get("error", "gateway declined the refund"))
    return result
