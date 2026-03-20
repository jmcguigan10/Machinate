class Machinator < Formula
  include Language::Python::Virtualenv

  desc "Prompt-first control-plane CLI for ML workspaces and pipelines"
  homepage "https://github.com/jmcguigan10/Machinator"
  url "https://github.com/jmcguigan10/Machinator/releases/download/v0.5.0/machinator-0.5.0.tar.gz"
  sha256 "4abd3c85ac1de8b2af78371d4ea6a0fb6df4bea440dab27549223c23598b8d8b"
  license "MIT"

  depends_on "python@3.12"

  def install
    virtualenv_install_with_resources
  end

  test do
    assert_match "Machinator", shell_output("#{bin}/macht --help")
  end
end
